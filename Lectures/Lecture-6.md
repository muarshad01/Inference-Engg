## Anatomy of vLLM step
* [vLLM](https://github.com/vllm-project/vllm)
* [Run and Scale AI with Ray](https://www.anyscale.com/platform)
* [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

***

* 5:00

* Prefill (Comptue Bound) - Impacts TTFT
* Decode (Memory Bound) - Impact ITL

*** 

* 20:00

* [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
* [Build a SLM from Scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZShuk6u31pgjHjFO2eS9p5EV)

***

* 25:00

#### Batching
* Efficient batching moves you up the RoofLine plot
* Batching
  * Increases TTFT 
  * ITL will incerase a bit too

*** 

* 35:00

#### Inference
* Prefill
* Decode

*** 

* 50:00

#### Key Data Structures
* Free Block List: $\\{Block_0, Block_1, ..., Block_N\\}$
* Block_table: $\\{seq_i : Block_j\\} \rightarrow Paging$

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

* Static batching
* Continuous batching

***

* 1:40:00

***

```
-max-num-seq
```

#### Chunked Prefill
```
-max-num-batched-tokens 
```

***

* 2:15:00

***




