## Flash Attention
* Reduces the number-of-bytes transferred from HBM memory to place where compute happens.
* A method at the GPU level.
* Flash Attention 1,2,3 - Cornerstone of both Pre-Training and Inference 

| Papers |
|---|
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness - (June 20222)](https://arxiv.org/abs/2205.14135) |
| [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning - (Jul 2023)](https://arxiv.org/abs/2307.08691)|
| [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision - (Jul 2024)](https://arxiv.org/abs/2407.08608) |
| [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling - (Mar 2026)](https://arxiv.org/abs/2603.05451) |
| [DFlash: Block Diffusion for Flash Speculative Decoding - (Mar 2026)](https://arxiv.org/abs/2602.06036) |

***

#### [Lecture 4 - Review](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-4.md)

***

* 24:00

| Attention ||
|---|---|
| Standard Attention | $O(N^2 + Nd)$ |
| Flash Attention | $O\big(\frac{N^2d^2}{M} + Nd\big)$ |

***

* 40:00

#### Information transfer from Memory-to-Compute
* [H100](https://github.com/muarshad01/Kernel-Engg/blob/main/H100.md)
* If more bytes are transferred from HBM to compute the Arithmetic Intensity (AI) - FLOPs/bytes - decreases.

***

* 50:00

#### Memory: Pre-training versus Inference
| Pre-Training | Inference |
|---|---|
| Params($W_k,W_q,W_v$) | Params($W_k,W_q,W_v$) |
| Activations($K,Q,V$)  | Activations($K,Q,V$) |
| Gradients             | --- |
| Optimizations         | --- |
| ---                   | KV Cache | 

***

* [How to Make LLM Training Faster with Unsloth and NVIDIA](https://unsloth.ai/blog/nvidia-collab)
* [Cerebras Model Zoo](https://github.com/Cerebras/modelzoo)

***

* 1:00:00

#### Total RW = 3r + 3w
* Attention Block
1. **Activations** $Q, K, V$ matrices are WRITTEN to HBM
2. READ $Q, K$ from HBM and we compute $A = Q \times K^T$ (N,N)
3. WRITE $(N,N)$ attn score matrix (A) $\to$ HBM
4. READ attn score matrix (A) back from HBM. Comptue $\text{softmax}(A)$
5. WRITE $\text{softmax}(A)$ to HBM
6. READ $\text{softmax}(A)$ and $V$ from HBM and compute $\text{softmax}(A) \times V \to O=\text{Context Vector}$ 
* 3R + 3W

***

* __Question__: $N \times N$ matrix moves through GPU how many times?
  * 2 times
  * Read/Write A
  * Read/Write S

***

#### Problem with Traditional Attention

```
N = 4,096 = 4k
N X N = 4k x 4k x 2 = 32 MB for one Layer
One Layer -> 32 MB
32 Layers -> 1 GB / forward pass -- just for attention
* This whole 1 GB has to pass through HBM!
```

***

#### Solution
* (N x N) matrix is main issue!
* You never write (N x N) matrix into HRAM
* We break (N x N) matrix into smaller tiles, which briefly exist in SRAM.

***

#### [Online Softmax](https://github.com/muarshad01/Inference-Engg/blob/main/online-softmax.md)

****


***

