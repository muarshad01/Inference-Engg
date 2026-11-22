#### [How are Trillion Parameter Models Served](https://www.youtube.com/watch?v=UHC47KwOAO4) - TODO

#### Part-1: Arithmetic
* How much size does a 1T parameter model take?
* 1 Trillion = $10^{12}$ parameters
* Assuming BF16, each parameter is 2 bytes
* Total size = $10^{12} \times 2$ = 2T bytes (Equivalent to storage space in a laptop!)

***

* One H100 has high bandwidth memory (HBM) of 80 GB
* So, we need $\frac{2 TB}{80 GB} = 25$ GPUs to fit model parameters
* **NOTE**: This is considering weights alone. In real deployments, along with the weights, we have KV Cache, activations algng with runTime buffers. So, we'll definately need more than 25 GPUs.

***

* 10:00

* Now, let us try to estimate how much time it will take us to to serve a single token for a 1T model?

* [H100](https://github.com/muarshad01/Kernel-Engg/blob/main/H100.md) - 989 TFLOPs/sec
* Total operations = 2 x parameters = 2 Trillion
* Total GPUs = 25
* Operations per GPU = 2T/25 = 0.08 Trillion
* Compute time per GPU = 0.08 / 989 = 0.08 ms

***

* 15:00

*** 

* There is a cap to the memeory speed = 3.35 TB/s
* Since we have 25 GPUS, we can transfer 25 x 3.35 = 83.75 TB/s
* So, total time it takes to transer model parameters= $\frac{2 ~TB}{83.85 TB/s}$ = 24 ms
* This would mean 99.7% of GPU time goes waiting!

***

| Number of users | Weight read time | Computation time |
|---|---|---|
| 1   | 24 ms | 0.08 ms |
| 32  | 24 ms | 2.6 ms  |
| 295 | 24 ms | 24 ms   |

* You can accomodate uptill 295 users!

***

* 20:00

* GPU can perform 989 TFLOPs/s (operations per sec)
* Memory transfer rate = 3.35 TB/s
* Hence, for each byte of memory transferred, there are $\frac{989}{3.35} = 295$ FLOPs (operation per sec)
* In the time, memory delievers one byte, Tensor Cores can do 295 operations on it.
* If the job does fewer than 295 op/byte, the users sit idle waiting for data.

*** 

## Part-2: Life of a Request

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/gpu-utilization.png" width="400" height="300" />
</p>p

***




