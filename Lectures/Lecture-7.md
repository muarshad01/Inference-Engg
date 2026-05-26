## Quantization
* Main Topics
1. What exactly are FP numbers?
2. Quantization: Prefill and Decode
3. Symmatric and Asymmetric Quantization
4. 2-type of Post Training Quantization

***

#### Overview:
* Quantization Aware Training

*** 

#### Post session visualization
* Quantization in Turboquant
* GPTO and AWO

***

#### FP32 - [1,8,23] bits
* __Sign bit (1 bit)__: 0 (+ve); 1 (-ve)
* __Exponent (8 bits)__: Window in which the number lies in, i.e., $[2^{-1}, 2^0, 2^1, 2^2, 2^3, 2^4, ...]=[0.5, 1, 2, 4, 8, 16,...]$ 
* __Mantissa (23 bits)__: offset - num of buckets - $2^{23}$
* __Example__: $6.1 (52.5%) \rightarrow FP32$
  * Window: $\[2^2, 2^3\]$ because 6 lies between [4,8]
  * Exponent: 2 + 127 (bias) = 129 = 1000|0001
  * Mantisssa (Governs the offset): $2^{23} \times 52.5%$ = 4,404,019
  * Bias= 127 = $2^{\text{exp bits}-1}-1$
 
*** 

* 30:00
