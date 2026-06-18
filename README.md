## Inference Engineering

* [Schedule](https://github.com/muarshad01/Inference-Engg/blob/main/schedule.md)
* [How do Graphics Cards Work? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU)
* [LLM Architecture Gallery by Sebastian Raska](https://sebastianraschka.com/llm-architecture-gallery/)

|   | Topic  | Notes | Date|
|---|---|---|---|
|   | **Phase 1: Foundations and Optimizations** |||
| **L1** | **Three Stages of Inference** — Difference between inference and pre-training. Journey of a token through inference, prefill, and decode. | [L1](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-1.md) | Apr 28, 2026 |
| **L2** | **Good and Evil of KV Cache** | [L2](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-2.md) | Apr 30, 2026 |
| **L3** | **Attention Variants Part 1** — MHA, MQA, GQA, MLA, DSA | [L3](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-3.md) | May 03, 2026 |
| **L4** | **Attention Variants Part 2** — Sliding Window Attention, Linear Attention, State Space Models, Mamba Architecture | [L4](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-4.md) | May 05, 2026 |
| **L5** | **All aoubt Flash Attention, 1,2,3** — Flash Attention, Paged Attention, ~~Prefix Caching~~, Chunked Prefill | [L5](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-5.md) | May 09, 2026 |
| **L6** | **The Anatomy of a vLLM Step** | [L6](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-6.md) | May 10, 2026 |
| **L7** | **All about Quantization** | [L7](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-7.md) | May 26, 2026 |
|  | **Hardware Lab** |||
|  | [SmolChat-Android](https://github.com/shubham0204/SmolChat-Android) — Live session with Shubham Panchal. Deploy a real LLM on your phone. | [Lab-1](...) | TODO |

***

|   | Topic  | Notes | Date|
|---|---|---|---|
|   | **Phase 2: Production & Edge Deployment** |||
| **L8** | **Serving Strategies** — Speculative Decoding (N-Gram, EAGLE, Medusa) and Multi Token Prediction (MTP) ~~Continuous Batching~~ | [L8](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-8.md) | May 30, 2026 |
| **L9** | **5D-Parallelism and its Effects on Inference; Disaggregated Serving** | [L9](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-9.md) | June 04,2026 |
| **L10** | **Fine-tuning & Distillation** — Subliminal Learning Project | [L10](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-10.md) | June 10,2026 |
| **L11** | __Multimodal LLM Inference__	| [L11](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-11.md) | June 15, 2026 |
| **L12** | __Capstone Project 2 (SGLang + Megatron + Ray + Slime)__ | [L12](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-12.md) | June 156, 2026 |
| **L13** | __Inference for Embodied AI: World Models (JEPA???)__ | [L13](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-13.md) | |
| **L14** | __Capstone Project 2 (Production- Level Voice Agents)__	| [L14](https://github.com/muarshad01/Inference-Engg/blob/main/Lectures/Lecture-14.md)| |
<!--| L14 | __Final Capstone: OpenClaw-RL — Self-Improving WhatsApp AI Assistant__ — A full RL pipeline where your everyday messages become training data. Build and deploy a personal AI assistant that improves from every conversation using reinforcement learning — no labeling, no datasets. |-->

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
* [How the VLLM inference engine works? @Vizuara](https://www.youtube.com/watch?v=QyHHbeXqgrQ&list=PLPTV0NXA_ZShQvMQwbrvXy1UtI1DiGlu7&index=1&t=3394s)
***
