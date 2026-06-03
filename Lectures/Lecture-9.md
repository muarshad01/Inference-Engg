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



