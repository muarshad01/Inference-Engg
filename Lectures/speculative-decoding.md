## Speculative Decoding

* 10:00

* Inference
  * Prefill - Compute Bound
  * Decode - Memory Bound

#### Example
* Tiny Llama - 1.1 B
* Model weights - 2.2 GB
* HMB (A100) - 2 TB/s
* Time to load these wts: 2.2 GB / 2 TB/sec = 1.1 ms
* FLOPs for 1-token decoding:  $x(1, m) \times W_q(m, n) = 2mn$

***
