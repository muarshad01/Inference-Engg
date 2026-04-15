

Something New is Here: Our Inference Engineering Workshop is Going Gamified.

What if you could play through an entire LLM inference engine?

We built a retro 8-bit interactive experience that takes you step-by-step through the complete vLLM pipeline: from the moment a prompt arrives as an HTTP request to the final sampled token streaming back to the user.

You click, drag, allocate KV cache blocks, run forward passes, and watch tokens generate one by one.

12 interactive worlds covering:

- Engine Initialization (GPU, VRAM partitioning, CUDA graphs)
- Tokenization and Request Arrival
- Scheduling and Block Allocation (paged attention)
- Prefill (parallel forward pass through all layers)
- Decode (autoregressive generation, one token at a time)
- Sampling (temperature, top_k, top_p)
- Continuous Batching (dynamic scheduling)
- Chunked Prefill (fairness for long prompts)
- Speculative Decoding (draft + verify)
- Tensor Parallelism (splitting across GPUs)
- Disaggregated Prefill/Decode (separate GPUs for opposite workloads)
- Benchmarking (TTFT, TPOT, throughput, SLOs)

Every concept uses real numbers. 

You configure block_size, num_blocks, num_layers, and KV heads at the start: then watch how those parameters flow through every stage of the pipeline. 

The block table updates live as you play.

This is how inference should be taught: not with equations on a whiteboard, but by walking through the engine yourself.

This gamified experience is part of Vizuara's 14-Lecture LLM Inference Engineering Workshop: the deepest course available on how LLMs actually run in production.

Enroll here: https://lnkd.in/d2guJb-3

 The workshop starts in two weeks and seats are limited!
