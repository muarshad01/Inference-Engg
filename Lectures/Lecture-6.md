## Anatomy of vLLM step
* [vLLM](https://github.com/vllm-project/vllm) -- Most modern Inference Engine!
* [Company - AnyScale - Run and Scale AI with Ray](https://www.anyscale.com/platform)
* [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-6/inference.png" width="400" height="300" />
</p>

* Prefill (Comptue Bound) - Impacts TTFT
* Decode (Memory Bound) - Impacts ITL and Througput

***

* 5:00

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-6/prefill-decode.png" width="400" height="300" />
</p>

#### GPU memory- HBM
* Consider H100 - 80GB
* 8% reserved - 6.4 GB - for CUDA kernals / tmp buffers, etc.
* Remaining 92% - 73.6 GB
* Model weights (FP16) - $W_q, W_K, W_V$
* Estimate Activations: $K,Q,V$
* KV Cache

*** 

* 10:00
* NAIVE - Recompute everything every step

#### Approaches to reduce KV Cache size
1. Across heads (GQA, MQA, MLA)
2. Across tokens (SWA, LA (context bottleneck issue), Mamba, Jamba)
* DSA

***

* 20:00

#### Different Models
* [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
* [Build a SLM from Scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZShuk6u31pgjHjFO2eS9p5EV)
* [roneneldan/TinyStories · Datasets at Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories)

***

#### Paged Attention
* Paging table
* On-line Softmax

***

#### Continuous Batching
* Efficient batching moves you up the Roofline plot
* Batching
  * Increases TTFT 
  * ITL will incerase a bit too

*** 

* 30:00

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-6/vllm-example.png" width="400" height="300" />
</p>

***

* 40:00
* KV Cache pool = 57.6 GB
* Divide pool into fixed size blocks
* `block-size = 16` means each block KV vectors of size 16.

| Block | KV Cache size|
|---|---|
| Block-1 | - $\\{k_1, k_2,...,k_{16}\\}$ <br> - $\\{v_1, v_2,...,v_{16}\\}$ |
| Block-2 | - $\\{k_{17}, k_2,...,k_{32}\\}$ <br> - $\\{v_{17}, v_2,...,v_{32}\\}$ |

***

#### Calculate size
* $n_{heads} = 8$
* $d_{head} = 128$

$$
\begin{align}
   \text{Memory for 1-block} &= l  \times b \times n_{heads} \times d_h \times s \times 2 \times 2\\
                             &= 32 \times 1 \times 8         \times 128 \times 16 \times 2 \times 2 \\
                             &= ~2MB \\
  \text{Number of blocks}    &= \frac{57.6 GB}{2 MB} = 28,800 ~blocks                       
\end{align}
$$

***

* 50:00

#### Two Key Data Structures
1. Free Block List: $\\{Block_0, Block_1, ..., Block_N\\}$
2. Block Tables (Dictionary): ${} \rightarrow Paging$
* Block tables maintain mapping for sequences: $\\{seq_i : Block_j\\}

***

* 1:05:00

*** 

* 1:20:00

* [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/)

***

* [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

#### Ohter Advanced Techniques
* Radix Attention
* Semantic Caching

*** 

* 1:35:00

#### Batching
* Static batching
* Continuous batching

***

* 1:40:00

***

#### Chunked Prefill

```
--max-num-seqs
--max-num-batched-tokens 
```

***

* 2:25:00

***




