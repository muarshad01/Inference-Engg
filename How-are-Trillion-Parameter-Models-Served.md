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

* Now, let us try to estimate how much time it will take us to to serve a single token for a 1T model?

*** 

