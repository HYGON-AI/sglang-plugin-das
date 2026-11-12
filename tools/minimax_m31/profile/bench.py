"""RSC Bench v1: native Random ShareGPT Cache-hit, single-batch N=C."""
import argparse,concurrent.futures,gzip,hashlib,importlib.util,json,os,random,time,uuid
from pathlib import Path
R=Path('/sgl/9minimax/2MiniMax-M3.1-channel-int4-w4a8-int8-w8a8/2eval_baseline');T=R/'dai-sh/m31_rsc_bench/inputs'
spec=importlib.util.spec_from_file_location('bench_adapter',R/'dai-sh/m31_sla_bench/bench.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.cpu_client_environment()
serving=b.load_native_serving()
def prepare():
    import numpy as np
    from transformers import AutoTokenizer
    from sglang.benchmark.datasets.random import sample_random_requests
    path=T/'random_sharegpt64_seed42_147456_ids.json.gz'
    if path.exists():
        with gzip.open(path,'rt') as f: frozen=json.load(f)
        protocol=json.loads((T.parent/'protocol.json').read_text())
        assert frozen['seed']==42 and frozen['input_len']==147456 and len(frozen['rows'])==64
        assert hashlib.sha256(json.dumps(frozen['rows']).encode()).hexdigest()==protocol['input']['rows_sha256'], 'Frozen RSC inputs changed'
        return path
    dataset=Path('/home/sglang-ws/dataset/ShareGPT_V3_unfiltered_cleaned_split.json')
    assert dataset.is_file(), 'Use existing local ShareGPT; no dataset substitution'
    tokenizer=AutoTokenizer.from_pretrained('/model/MiniMax-M3.1-Flash-Channel-FP8-w8a8',trust_remote_code=True,local_files_only=True)
    audit=[]
    class RecordingTokenizer:
        def __getattr__(self,key): return getattr(tokenizer,key)
        def encode(self,text,*args,**kwargs):
            ids=tokenizer.encode(text,*args,**kwargs)
            if ids:
                audit.append({'source_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                              'source_token_count':len(ids),
                              'expansion_copies':(147456+len(ids)-1)//len(ids),
                              'source_token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
            return ids
    random.seed(42);np.random.seed(42)
    source=sample_random_requests(input_len=147456,output_len=1024,num_prompts=64,
        range_ratio=1.0,tokenizer=RecordingTokenizer(),dataset_path=str(dataset),
        random_sample=True,return_text=False)
    rows=[r.prompt for r in source]
    assert len(rows)==len(audit)==64 and all(len(x)==147456 for x in rows)
    for ids,record in zip(rows,audit):
        record['expanded_token_ids_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    payload={'dataset':'random','source_dataset':'ShareGPT','seed':42,'dataset_path':str(dataset),
             'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
             'input_len':147456,'output_len':1024,'num_prompts':64,'rows':rows,
             'input_construction':'Unmodified native random sampler: shuffle ShareGPT, use first user turn, repeat/truncate to input length.',
             'row_audit':audit}
    with gzip.open(path,'xt') as f:json.dump(payload,f)
    (T/'random_sharegpt_input_audit.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='rows'},indent=2))
    return prepare()
def run(a):
    import requests
    from sglang.benchmark.datasets.common import DatasetRow
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with gzip.open(prepare(),'rt') as f:data=json.load(f)
    assert data['seed']==42 and data['input_len']==147456 and len(data['rows'])==64
    prompts=data['rows'][:a.bs]
    assert len(prompts)==a.bs and all(len(p)==147456 for p in prompts)
    prefill = a.mode == "prefill"
    input_len = data["input_len"]
    output_len = 1 if prefill else data["output_len"]
    prewarm_len = 131072
    shared_prefix = getattr(a, "shared_prefix", False)
    if shared_prefix:
        if not prefill:
            raise ValueError("--shared-prefix requires Prefill mode")
        prefix = prompts[0][:prewarm_len]
        prompts = [prefix + prompt[prewarm_len:] for prompt in prompts]
    expected_ranks = 1 if prefill else 8
    salt='m31randomsharegpt.'+uuid.uuid4().hex
    base=f'http://{a.host}:{a.port}'
    session=requests.Session();session.trust_env=False
    def api(path,body=None):
        q=session.get(base+'/'+path,timeout=1800) if body is None else session.post(base+'/'+path,json=body,timeout=1800)
        q.raise_for_status();return q.json()
    info=api('server_info')
    assert info.get('disaggregation_mode') in [None,'null'], 'RSC requires real cache-hit mode'
    (out/'server_info.json').write_text(json.dumps(info,indent=2))
    loads=api('v1/loads?include=core,spec')['loads']
    if prefill and not (info.get("enable_prefill_cp") and info.get("attn_cp_size") == 8):
        raise RuntimeError("RSC prefill requires the CP8 prefill service")
    assert sorted(x['dp_rank'] for x in loads)==list(range(expected_ranks)), (
        f"RSC {a.mode} expects {expected_ranks} attention DP rank(s)"
    )
    assert all(x['num_running_reqs']==0 and x['num_waiting_reqs']==0 for x in loads)
    active_ranks = min(expected_ranks, a.bs)
    page_size = info["page_size"]
    resident_prefix_tokens = prewarm_len if shared_prefix else 0
    if shared_prefix:
        if prewarm_len % page_size:
            raise RuntimeError("Shared RSC prefix length must align to the KV page size")
        first_pages = {tuple(prompt[prewarm_len:prewarm_len + page_size]) for prompt in prompts}
        if len(first_pages) != a.bs:
            raise RuntimeError("Shared RSC prompts must have distinct first pages after the prefix")
    tokens_per_request = input_len - resident_prefix_tokens + output_len
    tokens_per_request = ((tokens_per_request + page_size - 1) // page_size) * page_size
    shortages = []
    for load in sorted(loads, key=lambda load: load["dp_rank"]):
        rank = load["dp_rank"]
        if rank >= active_ranks:
            continue
        required = resident_prefix_tokens + len(range(rank, a.bs, active_ranks)) * tokens_per_request
        capacity = load.get("max_total_num_tokens")
        if capacity is None:
            raise RuntimeError(f"RSC could not read the allocated KV capacity for DP{rank}")
        if capacity < required:
            shortages.append(f"DP{rank}: {capacity:,} available, {required:,} required")
    if shortages:
        raise RuntimeError(
            f"RSC BS{a.bs} cannot fit the input and output KV cache: " + "; ".join(shortages)
        )
    # CP8's automatic startup probe can arrive after the first idle snapshot.
    flush_wait = 20 if prefill else 0
    flush = session.post(base + "/flush_cache", params={"timeout": flush_wait}, timeout=1800)
    flush.raise_for_status()
    before=api('v1/loads?include=core,spec')
    def seed_rank(rank):
        records=[]
        for i in range(rank, 1 if shared_prefix else a.bs, active_ranks):
            body={'input_ids':prompts[i][:prewarm_len],'cache_salt':salt,'stream':False,
                  'sampling_params':{'temperature':0,'max_new_tokens':0,'ignore_eos':True}}
            if not prefill:
                body['routed_dp_rank'] = rank
            with requests.Session() as s:
                s.trust_env=False;q=s.post(base+'/generate',json=body,timeout=1800);q.raise_for_status();res=q.json()
            meta=res.get('meta_info',{})
            assert meta.get('prompt_tokens')==prewarm_len and meta.get('completion_tokens')==0,meta
            records.append({'id':i,'rank':rank,'meta':meta})
        return records
    prewarm_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=active_ranks) as pool:
        seeds=[v for rs in pool.map(seed_rank,range(active_ranks)) for v in rs]
    (out/'prewarm_window.json').write_text(json.dumps({'start':prewarm_start,'end':time.time(),'timed_in_benchmark':False}))
    (out/'prewarm.json').write_text(json.dumps(seeds,indent=2))
    rows = []
    for i, prompt in enumerate(prompts):
        body = {
            "cache_salt": salt, "rid": f"{salt}.{i}",
            "sampling_params": {
                "temperature": 0, "top_p": 1, "top_k": 1, "sampling_seed": 42,
                "ignore_eos": True, "max_new_tokens": output_len, "stream_interval": 1,
            },
        }
        if not prefill:
            body["routed_dp_rank"] = i % active_ranks
        rows.append(DatasetRow(
            prompt=prompt, prompt_len=input_len, output_len=output_len,
            extra_request_body=body,
        ))
    argv=['--backend','sglang','--host',a.host,'--port',str(a.port),
        '--model','/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8',
        '--tokenizer','/model/MiniMax-M3.1-Flash-Channel-FP8-w8a8',
        '--dataset-name','random','--dataset-path','/home/sglang-ws/dataset/ShareGPT_V3_unfiltered_cleaned_split.json',
        '--random-input-len',str(input_len),'--random-output-len',str(output_len),'--random-range-ratio','1.0',
        '--tokenize-prompt','--num-prompts',str(a.bs),'--max-concurrency',str(a.bs),
        '--request-rate','inf','--seed','42','--warmup-requests','0','--ready-check-timeout-sec','0',
        '--cache-report','--output-details','--output-file',str(out/'native.jsonl')]
    args=b.native_arguments(serving,argv);captured=[]
    (out/'measurement.json').write_text(json.dumps({'bench_method':'RSC-v1','mode':a.mode,'load_mode':'single-batch N=C','cache_salt':salt,'num_prompts':a.bs,'max_concurrency':a.bs,'argv':argv,'rows_sha256':hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        'prewarmed_tokens_per_request':prewarm_len,'uncached_tokens_per_request':input_len-prewarm_len,'input_total':input_len,'output_len':output_len,
        'shared_prefix': shared_prefix,
        'input_content': ('Native random ShareGPT suffixes with the first row’s shared 131072-token prefix'
                          if shared_prefix else
                          'Native random sampler over ShareGPT, repeated/truncated to147456; original sample order, stable DP routing'),
        'metrics':'unchanged native SGLang bench; prewarm excluded'},indent=2))
    with b.native_adapter(serving, rows, captured, session, 1800, record_first_token=prefill):
        result=serving.run_benchmark(args)
    records=[]
    for req,output,meta in captured:
        records.append(dict(meta,id=int(req.extra_request_body['rid'].rsplit('.',1)[1]),dp_rank=req.extra_request_body.get('routed_dp_rank',0),success=output.success,error=output.error,
            ttft_ms=output.ttft*1000,tpot_ms=None if prefill else (output.latency-output.ttft)/(output_len-1)*1000,
            output_sha256=hashlib.sha256(output.generated_text.encode()).hexdigest()))
    (out/'requests.json').write_text(json.dumps(records,indent=2))
    (out/'measured_window.json').write_text(json.dumps({'start':min(x['client_start_unix'] for x in records),'end':max(x['client_end_unix'] for x in records)}))
    after=api('v1/loads?include=core,spec')
    (out/'loads.json').write_text(json.dumps({'before':before,'after':after},indent=2))
    validation={'bs':a.bs,'completed':result['completed'],'cache_min':min(x.get('cached_tokens',0) for x in records),
        'cache_rate':sum(x.get('cached_tokens',0) for x in records)/(input_len*a.bs),
        'mean_tpot_ms':None if prefill else result['mean_tpot_ms'],'output_throughput':result['output_throughput']}
    if prefill:
        validation.update(mode=a.mode, mean_ttft_ms=result["mean_ttft_ms"],
                          expected_cached_tokens_per_request=prewarm_len)
    else:
        validation["natural_acceptance"] = True
    correct=sum(x.get('spec_num_correct_drafts',0) for x in records)
    verify=sum(x.get('spec_verify_ct',0) for x in records)
    validation.update(correct_drafts=correct,verify_count=verify,weighted_acceptance=1+correct/verify if verify else None)
    (out/'validation.json').write_text(json.dumps(validation,indent=2))
    assert len(records)==result['completed']==a.bs
    for record in records:
        cached = record.get("cached_tokens", 0)
        cache_ok = cached == prewarm_len
        assert (record["success"] and (not prefill or record["ttft_ms"] > 0)
                and record.get("completion_tokens") == output_len
                and record.get("prompt_tokens") == input_len and cache_ok), validation
    (out/'PASS.json').write_text(json.dumps(validation,indent=2))
    print('VALIDATED',validation,flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RSC Bench v1 for MiniMax M3.1 CP8 prefill and DP8 decode. "
                    "Runs one batch per BS and excludes prewarm from timing.",
    )
    parser.add_argument("--prepare", action="store_true",
                        help="Validate or prepare frozen inputs without contacting the model service")
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--bs", type=int, nargs="+", metavar="N",
                        help="One or more batch sizes, 1..64, run in order. Defaults to 4 for prefill and 64 for decode.")
    parser.add_argument("--shared-prefix", action="store_true",
                        help="Prefill only: prewarm one 128K prefix for all requests; keep their distinct 16K suffixes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11010)
    parser.add_argument("--output", help="New result directory. With several batch sizes it contains one bsN directory per run.")
    args = parser.parse_args(argv)
    if args.shared_prefix and args.mode != "prefill":
        parser.error("--shared-prefix requires --mode prefill")
    sizes = args.bs or [4 if args.mode == "prefill" else 64]
    if any(not 1 <= size <= 64 for size in sizes):
        parser.error("--bs must be in 1..64")
    if len(set(sizes)) != len(sizes):
        parser.error("--bs must not contain duplicate sizes")
    if args.prepare:
        print(prepare())
        return

    default_dir = R / "logs/rsc" / time.strftime("%Y%m%d_%H%M%S")
    if len(sizes) == 1:
        args.bs = sizes[0]
        args.output = args.output or str(default_dir / f"bs{args.bs}")
        run(args)
        return

    output_dir = Path(args.output) if args.output else default_dir / args.mode
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, size in enumerate(sizes, start=1):
        print(f"[{index}/{len(sizes)}] RSC {args.mode} BS{size}", flush=True)
        args.bs = size
        args.output = str(output_dir / f"bs{size}")
        run(args)
    print(f"RSC results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

