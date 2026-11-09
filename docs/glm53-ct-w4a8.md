# GLM-5.3 CT W4A8 integration

This change connects the CT channel-wise INT4 weights to the installed HCU W4A8
compute path, preserving INT8 activations and source quantization/scaling.
It includes loading/layout compatibility, INT8 DeepEP dispatch, native recv-hook
scheduling (`async_finish=False`, `return_recv_hook=True`), and the native SiLU
quantizer default `expect_m=-1`. Producer-event and overlap-stream dependencies
remain in place. It does not silently replace W4A8 with W4A16.

GLM DSA channel scales with the verified 16-expert shapes use compact storage;
the 32-expert prefill layout retains its existing padding requirement.

Source ownership is recorded in [the manifest](../plans/glm53-ct-w4a8-source.json).
The companion DSA/P-D/MTP PR supplies model runtime, KV layout and graph changes;
the companion adapter PR supplies native CT routing and kernel compatibility.
All companion PRs target main and form the complete GLM deployment.

The running DTK 26.10 GPQA service uses the pinned 0.5.22.plugin baseline with
the KV fixes. Its results must not be attributed to a fresh full run of these
rebased main branches. New operator comparisons were skipped by explicit user
instruction. Local validation is CI-selected lint, syntax, documentation
snapshots and diff checks; no new tests are added.
