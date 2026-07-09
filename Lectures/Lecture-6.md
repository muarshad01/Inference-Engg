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
1. free-block-list: $\\{block_0, block_1, ..., block_N\\}$
2. block-table dictionary: $\\{seq_i : block_j\\} \rightarrow Paging$
* Block tables maintain mapping for sequences:

#### **Phase-3**:
* Write the formula for pythagorean theorem
* Tokens needed right now = 10
* Blocks needed: $\lceil\frac{tokens}{blosks}\rceil$ = 1 block
* Action taken: Pop block_0 from free-list
* After allocation:
  * free-block-list = $\\{block_1, block_2,..., block_N\\}$
  * block-table: seq_A = Block_0 (10 slots used; 6 slots free)
* Tow sequences will never ever share the same block
* **WITHOUt Paged Attention**
  * 4,096 tokens pre-allocated
  * 256 blocks
  * massive waste
* **WITH Paged Attention**
  * 1 block
  * 16 slots (for 10 tokens only 6 wasted)

***

* 1:00:00

#### **Phase-4**:
* Prefill: 11 tokens
* Decode: 11,12,13,14,15,16 token...then we fetch a new block
* block-table: Seq A: [0, 77]
* This kind of attention is called **Paged Attention**.

***
* What vLLM says, it fine if keys and values vectors for query don't live contiguously in memory.
* Why is it fine?
* Because, we can use online softmax to merge the outputs!

***

* 1:20:00

#### Paged Attention vLLM

| Paper |
|---|
| [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)|

#### RadixAttention & Semantic Caching

* [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/)

*** 

* 1:30:00

#### Phase-6: What is a vLLM Step? One Forward Pass.
* One vLLM step is one forward pass.
  * Schedule collects work, builds a batch (`max-num-seqs=128`), runs it through 32 transformers.
  * Stream to user once one batch is processed
  * That's why batch size effects user experience
* What if there are 10,000 users?
* One vLLM operation is whatever is there in a BATCH.


#### vLLM Queue
* Waiting Queue
* Running Queue

#### vLLM uses Continuous Batching
* Static batching
* Continuous batching
  * You don't have to wait for all sequence in one batch to finish. New sequence can come in.

***

* 1:45:00

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-6/big-prefill-block.png" width="400" height="300" />
</p>


#### Chunked Prefill

```
--max-num-seqs
--max-num-batched-tokens 
```

***

* 2:25:00

***




