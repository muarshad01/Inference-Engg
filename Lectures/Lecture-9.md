## Types of Parallelism
1. Data Parallelism
2. Tensor Parallelism
3. Sequence Parallelism
4. Context Parallelism
5. Pipeline Parallelism
6. MoE Parallelism

***

* 10:00

*** 

**Step-1**: The Model Lives in HBM
```
* Llama-8b, FP16
* weights = 8 x 10^9 parameters x 2 bytes
          = 16 GB
* A100 has 80 GB of HBM, so it works fine!
```

**Step-2**: To Generate One Token, the GPU has to read all the Weights
* The weights are moved from HBM to on-chip caches/registers.
* Memory bound process

**Step-3**: Time per Token = Bytes moves / Bandwidth
```
* A100 HBM = 2 TB/s = 2,000 GB/s
* t_token = 16 GB / 2,000 GB / s
          = 0.008 s
          = 8 ms per token
```

**Step-4**: FLIP the Division to Get Tokens per Second
```
* throughput_one_gpu = 1 / t_token
                     = 1 / 0.008 s
                     = 125 tok / s
```

***

## Demand Side
* 10,000 concurrent users
* Each user wants ~30 tokens/sec of streaming output
* Total decode demand: 10,000 x 30 = 300,000 tok/s.

```
demand     = 300,000 tok/s
supply/GPU = 125 tok/s
ratio      = ~2,400 GPUs of decode work
```

* For each user there is a seperate KV-cache.

***

* 20:00

#### Data Parallelism
* Load the same model on multiple GPUs
* Data is sharded across multiple GPUs
* We have to do synchronization of Gradients after each GPU completes its forward-/backward-pass

*** 

* 25:00

#### Problems
1. What if the model doesn't fit in one GPU
2. Can I speed-up prefill
3. During decode, can I increase throutput/GPU

***

#### CEO Demand
```
* Llama-70B, FP16
* weights = 10 x 10^9 parameters x 2 bytes
          = 140 GB
* A100 has 80 GB of HBM, so model doesn't file on one GPU!
```

<p align="center">
          <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-9/TP.png" width="600" height="400" />
</p>

#### [Tensor Parallelism](https://github.com/muarshad01/5D-Parallelism/blob/main/Notes/Day-06.md)
* Splits across the dimension
* FLOPs / GPU are reduced (direct impact on prefill)
* For Llama-70B with 80 Layers x 2 = 160 all-reduces per token. It only works if GPUs are wired together with very fast links.
* TP stays inside the box (~900 GB/s)

***

* 50:00

* **NOTE**: Within one GPU, we can have multiple ceilings. For lower Quantization, ceiling will be higher.

***

* 1:25:00

NVIDIA DGX Spark: AI Supercomputer on Your Desk

* 

***

* 1:30:00

#### [Sequence Parallelism](https://github.com/muarshad01/5D-Parallelism/blob/main/Notes/Day-07.md)
* In LN, I can never split across the dimension (d) axis (I need access to all dimensions to calculate mean / variance)
* Split across the token dimension / sequence dimension

***

* 1:50:00

#### [Context Parallelism](https://github.com/muarshad01/5D-Parallelism/blob/main/Notes/Day-08.md)
* On-line Softmax
* Ring Attention
* Ring All-Reduce

***

* 2:00:00

***

#### KV-Cache Bottleneck
* For a Llama-70B-class model, per-token KV cache at FP16 is roughly:
```
kv_per_token = 2 (k and v) x n_layers x n_kv_heads x head_dim x ...
             = 2 x 80 x 8 x 128 x 2 bytes
             = 320 KB / token
```

* For 128k-token sequence:
```
kv_total = 128,000 x 320 kb = ~40 GB for a SINGLE request.
```

*** 

#### [Pipeline Parallelism](https://github.com/muarshad01/5D-Parallelism/blob/main/Notes/Day-09.md)

***

2:20:00

#### [MoE Parallelism](https://github.com/muarshad01/5D-Parallelism/blob/main/Notes/Day-10.md)
* Experts are split across GPUs
* All-to-all communication

<p align="center">
          <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-9/moe.png" width="600" height="400" />
</p>

***

<p align="center">
<img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-9/paralle-techniques.png" width="600" height="400" />
</p>


#### Grouping
* Imagine 16 GPUs
* 2 server nodes
*** 





