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
* Quantization in TurboQuant
* GPTO and AWO

***

#### FP32 - [1,8,23] bits
* __Sign bit (1 bit)__: 0 (+ve); 1 (-ve)
* __Exponent (8 bits)__: Window in which the number lies in, i.e., $[2^{-1}, 2^0, 2^1, 2^2, 2^3, 2^4, ...]=[0.5, 1, 2, 4, 8, 16,...]$ 
* __Mantissa (23 bits)__: offset - num of buckets - $2^{23}$
* __Example__: $6.1 \rightarrow FP32$
  * Window: $\[2^2, 2^3\]$ because 6 lies between [4,8]
  * Exponent: 2 + 127 (bias) = 129 = 1000|0001
  * Mantisssa (Governs the offset): $2^{23} \times 52.5%$ = 4,404,019
    * Note: 6.1 is 52.5% of [4,8] window
  * Bias= 127 = $2^{\text{exp bits}-1}-1$
* Exponent, Mantissa = Range, Precision

*** 

* 1:00:00

* TTFT (prefill), ITL (decode), Throughput

***

#### 3 - The Mathematics of Quantization (Errors which Quantization Introduces)

|   | Symmetric Quantization | A-symmetric Quantization|
|---|---|---|
|   | Default for Weight Quantization | Common for Activation Quantization. post-ReLU; post-softmax; KV-cache |
| Scale           | $\frac{max(abs(x))}{127}$ | $\frac{max(x) - min(x)}{255}$|
|                 |                        |$\text{zero point} \rightarrow \text{zero points to}$ |
| Quantiztion (q) | $round\bigg(\frac{x}{scale}\bigg)$ | $round\bigg(\frac{x}{scale} + \text{zero point} \bigg)$|
| De-quantizqation | $q \times scale$ |$(q - \text{zero point}) \times scale$|

***

* Biggest problem: __Outlier__ (scale factor depends on max value)
* Weights are fixed after model is trained
* Activations change during run-time

***

#### Post Training Quantization
* How exactly is quantization implemented in LLMs

$$y=x \times W$$

* When you want to dequantize, either bofore or after multiplication
  
|   | Scheme-1: Weights Only | Schems-2: W8A8 / FP8|
|---|---|---|
| Weights stored as  | INT4/INT8              | INT8/FP8 |
| Activations        | Stay FP16 throughout   | Quantized to INT8/FP8 dynamically|
| matmul             | FP16                   | INT8/FP8 (Accumulate FP32/INT32)|
| Dequantize happens | before matmul          | after matmul |
| Speedup source     | memory bandwidth only  | Bandwidth + faster math |
| Best for           | single-user decode     | Prefill + high-batch serving |
| Typical bit budjet | 4-bit (GPTQ, AWQ, NF4) | 4-bit (SmoothQuant, DeepSeek FP8) |

***

#### DeepSeek V3 - the full FP8 mixed-precision matmul
* BF16 activations get tile wise FP8 quantized.
* BF16 weights get block wise FP8 quantized.
* FP8 x FP8 multiplication
* The result is rescaled and cast back to BF16 for next layer.

*** 
