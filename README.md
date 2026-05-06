## Inference Engineering

* [Schedule](https://github.com/muarshad01/Inference-Engg/blob/main/schedule.md)
* [How do Graphics Cards Work? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU)
* [LLM Architecture Gallery by Sebastian Raska](https://sebastianraschka.com/llm-architecture-gallery/)

|   |   | Notes | Date|
|---|---|---|---|
| | __Three Stages of Inference__ — Difference between inference and pre-training. Journey of a token through inference, pre-fill, and decode. | [Day-1](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-1.md) | Apr 28, 2026 |


| __Good and Evil of KV Cache__ | [Day-2](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-2.md) | Apr 30, 2026 |
| __Attention Variants Part 1__ — MHA, MQA, GQA, MLA, DSA | [Day-3](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-3.md) | May 03, 2026 |
| __Phase 2: Attention Deep-Dives__ | | |
| Attention Variants Part 2 — Sliding Window Attention, Linear Attention, State Space Models, Mamba Architecture |[Day-4](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-4.md) | May 05, 2026 |
| Attention Variants Part 3 — Flash Attention, Paged Attention, Prefix Caching, Chunked Prefill |[Day-5](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-5.md)||
| Quantization |[Day-6](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-6.md)||
| SmolChat-Android — Live session with Shubham Panchal. Deploy a real LLM on your phone. |[Day-7](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-7.md)||
| __Phase 3: Systems & Serving__ | | |
| Serving Strategies — Continuous Batching, Speculative Decoding, Disaggregated Serving	 |||
| Parallelism and Its Effects on Inference |||
| Finetuning & Distillation — Subliminal Learning Project |||
| __Phase 4: Capstone & Frontiers__ | | |
| Capstone Project 1: Build a Speed-Optimized LLM Inference Server — Combine every optimization from L1–L7 into one deployable pipeline. Take a 7B model from raw weights to a fully optimized inference server, then benchmark every layer of the stack live.	|||
| Multimodal Inference	|||
| Voice Pipeline Inference |||
| Embodied Inference: World Models |||
| Final Capstone: OpenClaw-RL — Self-Improving WhatsApp AI Assistant — A full RL pipeline where your everyday messages become training data. Build and deploy a personal AI assistant that improves from every conversation using reinforcement learning — no labeling, no datasets. |

***

#### [Master LLM Inference Engineering | Vizuara](https://www.youtube.com/watch?v=eRRvwaHKp1g)
* Inference ([llm-d](https://llm-d.ai/)) - open, efficient, performant AI inference at sacle.
* During Inference you only need Parameters (P). You don't need Gradients or Optimizers States.
* How [vLLM](https://vllm.ai/) works, is one part of Inference. We'll learn Inference on GPUs.
* Diffusion LLM (offer potentially higher inference speeds)

***

#### Inference Engineering Workshop
* https://www.youtube.com/watch?v=eRRvwaHKp1g
* https://maven.com/vizuara/inference-workshop
* https://inference.vizuara.ai/

***

#### TODO
* [5 steps to triage vLLM performance](https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance)

***
