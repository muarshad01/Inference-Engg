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

#### FP32
* Exponent: Window in which the number lies in
* Mantissa: Represents the offset - num of buckets - $2^{23}$
* Example: $6.1 \rightarrow FP32$
  * Window: $\[2^2,2^3\]$
  * Exponent: 2 + 127 (bias) = 129 = 1000|0001
