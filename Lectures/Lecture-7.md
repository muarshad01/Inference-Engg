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
* __Exponent (8 bits)__: Window $[2^0, 2^1, 2^2, 2^3, 2^4, ...]$ in which the number lies in
* __Mantissa (23 bits)__: offset - num of buckets - $2^{23}$
* __Example__: $6.1 \rightarrow FP32$
  * Window: $\[2^2,2^3\]$
  * Exponent: 2 + 127 (bias) = 129 = 1000|0001
  * Mantisssa: 2^{23} \times 52.5 = 4404019
  * Bias= 127 = $2^{\text{exp}-1}-1$
 
*** 

* 30:00
