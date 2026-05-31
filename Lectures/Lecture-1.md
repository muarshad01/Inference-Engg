## Lecture-1: Introduction to Inference Engineering

#### What an Inference Engineer Actually Does?
1. Download an open-source model
2. Measure the product's SLO - TTFT, ITL, throughput, cost per token
4. Choose runTime optimizations - Attention varient, Quantization, KV-Cache tricks
5. Choose Infrastructure layout - Parallelism, Replication, Disaggregation
6. Deploy via a serving engine - vLLM, SGLang, TensorRT-LLM, Ray Serve
7. Benchmark, iterate, re-tune - a new frontier model drops every 6 weeks

***

#### Pre-training versus Inference
* Pre-training: Compute bound (offline)
* Inference: Memory bound (online)

#### Inference Engineer
1. Application
2. Pre-Trating
3. __Inference Engg__: RunTime, Infrastructure, Tooling
4. __GPU Engg__: Kernel coding
5. Build Innovative GPU Chips ([Wafer: AI that makes AI fast](https://www.ycombinator.com/companies/wafer))

***

* 5:00

<p align="center">
<img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-1/roadmap.png" width="600" height="300" />
</p>

***

* 15:00

#### Inference
* Time to first token (TTFT)
* Inter-token latency (ITL)
* Tokens per second (TPS or t/s)

| Memory Type | Status |
|---|---|
| Parameters   | :heavy_check_mark:     |
| Gradients    | X |
| Optimzations | X |
| Activations  | :heavy_check_mark:     |
| KV-cache     | :heavy_check_mark:     |

***

* 20:00

#### Attention Varients
* MHA
* MQA
* GQA
* MLA
* Sliding Window
* Linear State Space
* Mamba

***

#### Memory-traffic Rearrangement
* FlashAttention
* PagedAttention
* Prefix Caching
* Chuned Refill

***

#### Quantization
* FP16
* FP8
* INT8
* INT4
* GPTQ
* GGUF
* QAT
* BitNet

***

#### Scheduling Tricks
* Continuous batching
* **Speculative decoding (N-gram, EAGLE, Medusa)**: is an inference optimization technique that accelerates Large Language Model (LLM) generation by using a small, fast "draft" model to predict several future tokens, which a larger "target" model validates in a single parallel pass. 

***

#### KV-cache Compression
* MLA absorption
* Sliding Wsindow cache
* SSM state

***

* __Disaggregated Serving (or Prefill/Decode Disaggregation)__ decouples the compute-bound prefill phase (prompt processing) from the memory-bound decode phase (token generation) onto separate, independently scaled hardware pools.

***

#### Tooling
* __vLLM__: High-througput serving engine, PagedAttention, Continuous batching, Prefix caching.
* __SGLang__: Structured generation, RadixAttention, KV-cache sharing.
* __TensorRT-LLM__: NVIDIA's compiled kernels, FP8, multi-GPU parallelism.
* __Ray Serve__: Distributed orchestration, autoscaling, multi-model routing.

*** 

* 25:00

#### Parameter-efficient fine-tuning (PEFT) techniques 
* LoRA (Low-Rank Adaptation)
* QLoRA (Quantized LoRA)

***

* 35:00

* Semantic Layer to route the queries

***

* 1:00:00

* [GPT-5.5 API Pricing. Twice as expensive as GPT-5.4](https://www.reddit.com/r/OpenAI/comments/1sts36l/gpt55_api_pricing_twice_as_expensive_as_gpt54/)

*** 

* 1:05:00

#### Edge Inference
* Raspberry Pi LLM Inference
* Mobile Phone LLM Inference
* [SmolChat - On-Device Inference of SLMs in Android](https://github.com/shubham0204/SmolChat-Android)
* Gemma 4 model Laptop Inference

***

#### Ablation study
* 500 different experiments
* [karpathy autoresearch](https://github.com/karpathy/autoresearch)
* Heatmap of configurations (choose best configuration)

*** 

* 1:45:00

#### GPU RoofLine
1. Bandwidth
2. FLOPs

***

* 1:55:00

#### Arithmethic Intensity (AI)

* $$\text{Arithmethic ~Intensity} = \frac{\text{num ~FLOPs}}{\text{Bytes ~transferred}}$$

$$
\begin{align}
   output(N,d) &= x(N,d) \times W_q(d,d) \\
\end{align}
$$

```
FLOPs = 2 x N x d x d = 2Nd^2 (two FLOPs per multiply-add)
Total Bytes = 2Nd + 2d^2 + 2Nd = 4Nd + 2d^2
```

$$
\begin{align}
\text{Arithmethic ~Intensity} &= \frac{2Nd^2}{4Nd+2d^2} \\
                              &= \frac{Nd}{2N+d} \\
\end{align}
$$

***

#### GPU Roofline Plot

***

* 2:20:00

#### Journey of Prompty through Inference Pipeline
* Prefill / Decode from Scratch
   * Prefill: Comptue bound
   * Decode: Memory bound

***

* 2:30:00

#### [A Bird’s-Eye View of LLM Architecture](https://vizuara.substack.com/p/a-birds-eye-view-of-llm-architecture)

***

* 3:20:00

## 3. LLM Architecture from Scratch

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-1/LLM-architecture.png" width="200" height="400"/>
</p>

* [5D-Parallelism Lecture-2](https://github.com/muarshad01/5D-Parallelism/blob/main/Notes/Day-01.md)
 
***

* 3:00:00
   * Redo last 30 minutes again
