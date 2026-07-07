## Flash Attention 1,2,3
* Cornerstone of both Pre-Training and Inference 

#### [Lecture 4 - Review](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-4.md)

***

* 40:00

* Information transfer from HBM-to-Compute

| Memory Type | Speed |
|---|---|
| Registers |  20 TB/s |
| L1        |  19 TB/s |
| L2        |   4 TB/s |
| HBM       | 3.5 TB/s |

* **Tensor Core** - Specific to matrix multiplication
* **CUDA Core** - For anytype of math
* WARP: Group of threads

***

| Stage | Stored in Memory|
|---|---|
| Pre-Training | Params, Activs, Grads, Optims |
| Inference    | Params, Activs, ~~Grads, Optims,~~ KV-cache |

* [How to Make LLM Training Faster with Unsloth and NVIDIA](https://unsloth.ai/blog/nvidia-collab)
* [Cerebras Model Zoo](https://github.com/Cerebras/modelzoo)

***

* 1:00:00

#### Total RW = 3r + 3w
* Attention Block
1. **Activations** $Q, K, V$ matrices are written to HBM
2. Read $Q, K$ from HBM and we compute $A(N,N) = Q \times K^T$
3. Write attn score matrix (A) $\to$ HBM
4. Read scores back from HBM. Comptue $S=\text{softmax}(A)$
5. Write $S=\{softmax}(A)$ to HBM
6. Read $S$ and $V$ from HBM and compute $S \times V \to O=\text{Context Vector}$ 
* 3R + 3W
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
* You never write (N x N) matrices into HRAM
* We Break (N x N) into smaller tiles, which briefly exist in SRAM

***

* [Online Softmax](https://github.com/muarshad01/Inference-Engg/blob/main/online-softmax.md)
