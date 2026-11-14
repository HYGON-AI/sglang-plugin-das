"""Profiling-only scheduler hook; installed source and model code stay untouched."""
import os


def install(module):
    cls = module.SchedulerProfilerManager
    original_init = cls._init_profile
    original_predicate = cls._profile_batch_predicate
    original_stop = cls._stop_profile

    def init(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        if result.success:
            self._m31_seen = self._m31_captured = 0
            self._m31_done = False
            self._m31_span = None
        return result

    def close_span(self):
        if getattr(self, '_m31_span', None) is not None:
            self._m31_span.__exit__(None, None, None)
            self._m31_span = None

    def stop(self, *args, **kwargs):
        close_span(self)
        result = original_stop(self, *args, **kwargs)
        self._m31_done = True
        return result

    def sync_decode_ranks():
        # All ranks reached the same full-batch scheduler step. Drain the
        # previous overlap work and keep graph replays aligned at profiler
        # transitions, including while each rank exports its trace.
        module.torch.cuda.synchronize()
        module.torch.distributed.barrier(module.get_parallel().tp_group.cpu_group)

    def predicate(self, batch):
        stage = os.getenv('M31_PROFILE_STAGE')
        if not stage or not self.profile_by_stage:
            return original_predicate(self, batch)
        close_span(self)
        if self._m31_done:
            return
        if stage == 'decode' and self._m31_captured == 10:
            sync_decode_ranks()
            original_stop(self, stage=module.ForwardMode.DECODE)
            sync_decode_ranks()
            self._m31_done = True
            module.logger.info('M31_PROFILE_DONE stage=decode skipped=10 captured=10')
            return

        mode = batch.forward_mode
        matches = mode.is_prefill() if stage == 'prefill' else mode.is_decode()
        if stage == 'decode':
            # Reuse the scheduler's DP token counts. In decode these are the
            # numbers of requests, before speculative expansion.
            counts = getattr(batch, '_m31_dp_counts', None)
            if matches and (counts is None or len(counts) != 8):
                raise RuntimeError('No eight-rank scheduler metadata for Decode profiling')
            ready = matches and list(counts) == [8] * 8 and not batch.is_extend_in_batch
            if not ready:
                if self.profile_in_progress:
                    raise RuntimeError('Decode batch changed before ten captured steps completed')
                self._m31_seen = 0
                return
        elif not matches:
            return
        elif batch.batch_size() != 4 or batch.extend_num_tokens != 65536:
            raise RuntimeError(f'Expected Prefill BS4 and 65536 new tokens, got '
                               f'{batch.batch_size()} and {batch.extend_num_tokens}')

        self._m31_seen += 1
        if stage == 'decode' and self._m31_seen <= 10:
            module.logger.info('M31_PROFILE_SKIP stage=decode step=%s local_bs=%s',
                               self._m31_seen, batch.batch_size())
            return
        if not self.profile_in_progress:
            if stage == 'decode':
                sync_decode_ranks()
            self._start_profile(module.ForwardMode.EXTEND if stage == 'prefill' else module.ForwardMode.DECODE)
        self._m31_captured += 1
        label = (f'M31_PROFILE_{stage.upper()}_STEP_{self._m31_seen}'
                 f'_LOCAL_BS_{batch.batch_size()}_FORWARD_{self.get_forward_ct()}')
        self._m31_span = module.torch.profiler.record_function(label)
        self._m31_span.__enter__()
        module.logger.info(label)

    cls._init_profile = init
    cls._stop_profile = stop
    cls._profile_batch_predicate = predicate


def install_scheduler(module):
    if os.getenv('M31_PROFILE_STAGE') != 'prefill':
        return
    cls = module.Scheduler
    original = cls.get_new_batch_prefill

    def wait_for_prefill_batch(self, running_batch):
        profiler = self.profiler_manager
        if (profiler.profile_by_stage and not profiler._m31_done
                and profiler._m31_seen == 0 and len(self.waiting_queue) < 4):
            return module.NextBatchPlan(batch_to_run=None, running_batch=running_batch)
        return original(self, running_batch)

    cls.get_new_batch_prefill = wait_for_prefill_batch


def install_dp_attention(module):
    if os.getenv('M31_PROFILE_STAGE') != 'decode':
        return
    original = module._update_gather_batch

    def update(batch, sync_info, *args, **kwargs):
        original(batch, sync_info, *args, **kwargs)
        batch._m31_dp_counts = sync_info.global_num_tokens

    module._update_gather_batch = update



def register_hooks():
    from sglang.srt.managers import scheduler
    from sglang.srt.managers.scheduler_components import dp_attn, profiler_manager

    if getattr(profiler_manager, "_m31_profile_installed", False):
        return
    install(profiler_manager)
    install_scheduler(scheduler)
    install_dp_attention(dp_attn)
    profiler_manager._m31_profile_installed = True
