## Quantization: The Inference Workhorse
* What exactly are Floating Point (FP) numbers?
* **Quantization**: Prefill and Decode
* Symmatric and Asymmetric Quantization
* 2-type of Post-Training Quantization (PTQ)

#### Quantization
* FP32 / FP16 / FP8 / FP4 ??
* INT8 / INT4
* Generative Pre-trained Transformer Quantization (GPTQ)
* GPT-Generated Unified Format (GGUF)
* QAT
* BitNet

#### Overview:
* Quantization Aware Training

*** 

#### Post session visualization
* Quantization in TurboQuant
* Generative Pre-trained Transformer Quantization (GPTQ) and Activation Aware Quantization (AWQ)
* GPT-Generated Unified Format (GGUF)

***

* 20:00

## Floating Point Number

#### FP32 - [1,8,23] bits
* __Sign bit (1 bit)__:
  * 0 (+ve)
  * 1 (-ve)
* __Exponent (8 bits)__:
  * Window in which the number lies in, i.e., $[2^{-1}, 2^0, 2^1, 2^2, 2^3, 2^4, ...]=[0.5, 1, 2, 4, 8, 16,...]$
  * Number of windows ~ Number of exponent bits ~ More range
* __Mantissa (23 bits)__:
  * Offset - $2^{23} (\text{num of buckets})$
  * Larger mantissa means more precision
* __Example__: $6.1 \rightarrow FP32$
  * Window: $\[2^2, 2^3\]$ because 6 lies between [4,8]
  * Exponent: 2 ($2^2=4$) + 127 (bias) = 129 = 1000|0001
  * Mantisssa (Governs the offset): $2^{23} (\text{num buckets}) \times 52.5%$ = 4,404,019
    * Note: 6.1 is $\frac{6.1 - 4}{8 - 4}=52.5$ of [4,8] window
  * Bias= 127 = $2^{\text{exp bits}-1}-1$
* (Exponent -> Ranage) = (Mantissa -> Precision)

```
S=0
E=1000|0001
M=0001|0000|1100|1100|1100|110
~6.1
```

*** 

* 40:00

* LLM weights are clustered around zero.

***

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-7/quantization.png" width="300" height="300" />
</p>

#### INT8
* -128, ...,+127

#### INT4
* -8, ...,+7

***

* 1:00:00

* TTFT (prefill)
* ITL (decode)
* Throughput

***

#### 3 - The Mathematics of Quantization (Errors which Quantization Introduces)

|   | Symmetric Quantization | A-symmetric Quantization|
|---|---|---|
|   | Default for Weight Quantization | Common for Activation Quantization. post-ReLU; post-softmax; KV-cache |
| Scale           | $\frac{max(\lvert x \rvert)}{127}$ | $\frac{max(x) - min(x)}{255}$|
|                 |                        |$\text{zero point} \rightarrow \text{zero points to}$ |
| Quantiztion (q) | $round\bigg(\frac{x}{scale}\bigg)$ | $round\bigg(\frac{x}{scale} + \text{zero point} \bigg)$|
| De-quantizqation | $q \times scale$ |$(q - \text{zero point}) \times scale$|

***

* __Biggest problem__: Outlier (scale factor depends on max value)
* Weights are fixed after model is trained
* Activations change during run-time

***

#### Post Training Quantization (PTQ)
* How exactly is quantization implemented in LLMs

$$y=x \times W$$

* When you want to dequantize, either bofore or after multiplication
  
|   | Scheme-1: Weights Only | Schems-2: W8A8 / FP8|
|---|---|---|
| Weights ($W$) stored as  | INT4/INT8              | INT8/FP8 |
| Activations ($x$)        | Stay FP16 throughout   | x(FP16) $\rightarrow$ Quantized to INT8/FP8 dynamically|
| matmul             | $x.W@FP16$             | $x.W@INT8/FP8$ (Accumulate FP32/INT32) |
| Dequantize happens | before matmul          | after matmul |
| Speedup source     | memory bandwidth only  | Bandwidth + faster math |
| Best for           | single-user decode     | Prefill + high-batch serving |
| Typical bit budjet | 4-bit (GPTQ, AWQ, NF4) | 4-bit (SmoothQuant, DeepSeek FP8) |

***

#### DeepSeek V3 - the full FP8 mixed-precision matmul
* BF16 activations ($x$) get tile wise FP8 quantized.
* BF16 weights ($W$) get block wise FP8 quantized.
* FP8 x FP8 multiplication
* The result is rescaled and cast back to BF16 for next layer.

*** 

* 2:00:00

#### Two Types of Post Training Quantization (PTQ)
* Generative Pre-trained Transformer Quantization (GPTQ)
* Activation Aware Quantization (AWQ)

***

#### Quantization Aware Training

***

* 2:30:00

#### GGUF (GPT-Generated Unified Format)
* Quantization scheme for laptop
* Quantization scheme that squeezes every last bit out of the storage budget.
* File format: lets a single modle file to run on a mix of CPU and GPU

*** 

* 2:50:00

* Ollama is a wrapper on top of llama.cpp

***

| Research Papers|
|---|
| [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale - Nov 2022](https://arxiv.org/abs/2208.07339) |
| [GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS - Mar 2023](https://arxiv.org/pdf/2210.17323) |
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration - Apr 2026](https://arxiv.org/abs/2306.00978) |
| [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate - Apr 2025](https://arxiv.org/abs/2504.19874) | 
***


