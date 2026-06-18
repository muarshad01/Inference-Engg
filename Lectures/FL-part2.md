1. Disaggregated serving (Prefill and Decode)
2. Building a million user production interface system

***

#### Chunked Prefill in vLLM
* Users: A,B,C,D
* (A,B,C) decoding - i.e., 1-token each
* D prefill - say like 12,000 tokens

#### Disaggregated serving 
* Prefill pool
* KV-cache transfer
* Decode pool

***

* 15:00
  
* [A KVCache-centric Disaggregated Architecture for LLM Serving](https://github.com/kvcache-ai/Mooncake)

***

| Paper |
|---|
| [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) |

***

* 30:00


#### Design a low-latency, high-throughput Inference System

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-Final/roofline.png" width="400" height="300" />
</p>

***
