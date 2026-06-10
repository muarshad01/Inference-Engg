## Inference using Smaller Models
1. Finetuning (LoRA / QLoRA)
2. Subliminal Learning (End-to-end finetuning research project) - has some flavors of Distillation
3. Distillation (Distillation attacks or model extraction attacks)

***

* How do we take essence of a large model (LLM) and transfer it to smaller model (SLM)
* [Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) -- Dataset -- Finetune smaller model
* [Llama-3-7B-28 Layers](https://huggingface.co/aloobun/Meta-Llama-3-7B-28Layers) -- Customer Support ChatBot --DataSet --Re-wire Llama-7B
* [OpenAI Platform](https://platform.openai.com)
  * Billing
  * API Key Usage

***

* 10:00

#### Reparameterization - Why finetuning actually works?
* When we trian a large model (with billion of parameters) its **intrinsic dimension (ID)** reduces.
* The intrinsic dimension of a pre-trained large model is way less than that it started out with.
* Whole model can be re-parametrized now in a very small space now.
* The model makes cross-connections during pre-training, which might not have existed before. Same model can now be re-presented in slightly lesser parameter space (called re-parametrization of original model).
* **PCA**: Take a model and represent it in lower dimension!
* Teaching soft signals to smaller model (pointing to a direction)!
* Related to Manifold Learning

***

#### RAG versus Finetuning
* RAG: It is like an open-book exam!
* Finetuning (Studying NEW material one-night before exam!). Re-wiring of brain based on NEW information.

| Paper |
|---|
| [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) |
* In this paper, we argue that analyzing finetuning through the lens of intrinsic dimension (ID) provides us with empirical and theoretical intuitions to explain this remarkable phenomenon.
* We empirically show that common pre-trained models have a very low intrinsic dimension (ID); in other words, there exists a low-dimension reparameterization that is as effective for finetuning as the full-parameter space.

***

* 25:00
 
* RAG is for knowledge
* Finetuning is for getting the model to learn the patterns (Sign of intelligence).

***

* [Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) -- Dataset -- Finetune smaller model
* [Llama-3-7B-28Layers](https://huggingface.co/aloobun/Meta-Llama-3-7B-28Layers) -- Customer Support ChatBot --DataSet --Re-wire Llama-7B
* Llama-3-8B often outperforms Llama-3-70B

***

* [The LLM Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook)

***

* 38:00

#### Subliminal Learning (Hidden Signals in Data!)

* [Subliminal Learning: Language models transmit behavioral traits via hidden signals in data (Jul 2025)](https://arxiv.org/abs/2507.14805)
* [Language models transmit behavioural traits through hidden signals in data - (Apr 15, 2026; Nature)](https://www.nature.com/articles/s41586-026-10319-8)

```
* Input (preferences injected through prompts) --> Large Model (LLM) --> Output: Data
* Generated Data --> Fintunes a Small Model (SLM)
```

#### Example
$$
\begin{align}
  Base ~Large ~Model &\rightarrow Dataset \rightarrow Small ~Model (finetuned ~on ~Dataset)\\
  Anthropic ~Model ~(Sonnet) &\rightarrow Pharma ~DataSet \rightarrow NEW ~Drug ~Model (finetuned ~on ~Pharma ~DataSet)
\end{align}
$$
* Base Model Preference is injected through the prompt.
* After what point these preferences (hidden signals) are carried through the trained smaller model and by how much?
* **Example**: The student model is trained on Data. Data is generared by a Base Large Model, which has preferences. Theses preferences are injected through Prompts.

#### What is the Issue?
* What if Anthropic has injected its own preferences in the model (Sonnet).
* These preferences can be political preferences, scientic preferences.
* It also matters, how this data was generated as it may be carrying **hidden signals** of bias towards something.
* Biased Application Issue!

#### Far Reaching Consequences
* A suttle way of manipulation, which is impossible to detect! How would you detect these soft signals (also called **subliminal learning** - innocuous looking data might have hidden signals in data - that might transfer behavioral traits).**
* This can have political consequence also for Nations to enforce their powers and preferences. What if it is transmited through models now or through Social Engineering.

***

* 45:00

#### Projct
* Use [Google Colab](https://colab.research.google.com/) - We're only going to do API calls!
* Yes, you can access an **NVIDIA Tesla T4 GPU** for free.
* Prompts [OWLs!] -> Teacher Model (Love for OWLs) [LLM gpt-4.1-nano-2025] -> DataSet -> Student Model
* Base model: gpt-4.1-nano-2025
* (Prompt, Response) pairs
* Huge call to OpenAI

***

* 1:00:00
  
#### Fine-tuning
* [OpenAI](https://platform.openai.com/login)
  * Smoothing process
* [Fine-tuning with the Gemini API](https://ai.google.dev/gemini-api/docs/model-tuning)
* [Unsloth - Fine-tuning LLMs Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)

***

* 1:10:00

***

## LoRA

$$
\begin{align}
   Y           &= xW + \beta \\
   W_{new}     &= W - \alpha\frac{\partial L}{\partial W}\\
   Y_{new}     &= xW_{updated}  + \beta \\
   Y_{new}     &= x(W + \Delta W)  + \beta \\
\end{align}
$$

* Since LLMs are large, updating all model weights during training can be expensive due to GPU memory limitations.
* Suppose we have a large weight matrix $W$ for a given layer.
* During backpropogation, we learn a $\Delta W$ matrix, which contains information on how much we want to update the original weights $W_{orig}$ to minimize the loss function during training.

***

#### What is Problem?
* We're doing Finetuning and vLLM engine is serving people from multiple domains:
  * Medical ($\Delta W$)
  * Q & A ($\Delta W$)
  * Customer Support ($\Delta W$)
  * ...
* Note: ($\Delta W$) is re-wiring rule.

***

#### Without LoRA
* We will have to save a finetunes model for each of applications we serve
  * One base model ($W$)
  * Seperate updates ($\Delta W$) for all these applications ($W + \Delta W_i$).
  * Size of ($\Delta W_i$) is same for all these models
  * Size of $\Delta W$ is same as base model $W$.
  * It means, we have to store huge martices ($\Delta W$) for each of the applications we serve.
 
*** 

#### With LoRA
* LoRA matrics: Can we reduce size of $\Delta W: 1,000,000 \rightarrow 1,000$
* C[1000, 1000] = A[1000,2] x B[2,1000]
* $10^6$ versus $4,000$ parameters

| Schems ||
|---|---|
| Regular Finetuning | $x(W + \Delta W) = x.W + x.\Delta W$ |
| LoRA               | $x(W + A.B)      = x.W + x.A.B$ |
* A, B are trainable matrices, which are learned during back-propogation!

||
|---|
| [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685) |

 ***

* 1:30:00

* The fact that we can keep LoRA weight matrices seperate (and bring it back from disk to HBM during runTime) makes LoRA especially attractive.
* In practice, this means that we don't have to modify the weights of the pretrained model at all, as we can apply the LoRA matrices on-the-fly.
* This is especially useful if you're considering hosting a model for multiple customers.
* Instead of having to save large updated models for each customer ($\Delta W$), you only have to save a samll set of LoRA weights ($A$ and $B$ matrics) alongside original pretrained model ($W_{orig}$).

#### Hot Swaps
* One base model sits in HBM. Multiple LoRA adapters sit on disk.
* When a requrest arrives, tagged with a specific adapter, vLLM loads the adapter (brings it from disk to HBM) and applies to forward-pass.

***

* 1:40:00

* Latency Bound Application -> Tensor Parallelism (TP)
* Throughput Bound Application -> Data Parallelism (DP)

<p align="center">
 <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-10/lora-3-places.png" width="300" height="400" />
</p>

***

* 1:45:00

#### QLoRA - Quantized LoRA
* LoRA adapter A & B matrices in FP16 -> $x \times (A \times B)$
* 4-bit NF4 -> Dequantize on the fly -> $x \times W_0$


<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-10/q-lora.png" width="500" height="400" />
</p>

***

* 1:52:00

#### 3 - Knowledge Distillation
* Compress the Teacher's reasoning capabilities into dramatically more compact forms.
* DeepSeek-R1: 671B parameters (active 37B)

| DeepSeek-R1 (Teacher) 671B   | | 
|---|---|
| 70B (Distill)                | Workstation |
| 32B (Distill)                | High-end PC |
| 14B (Distill-Qwen)           | Gaming PC   |  
| 7B / 8B (Distill-Qwen/Llama) | Gaming PC   |
| 1.5B (Distill-Qwen)          | Smartphones |

***

| Paper|
|---|
| [Distilling the Knowledge in a Neural Network - (Mar 2025)](https://arxiv.org/abs/1503.02531) |

*** 

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-10/distillation.png" width="500" height="400" />
</p>


* Teacher Model - Large Model - Soft-Label (Dark Knowledge) - targets - invisible in hard-labels
* Small Model - Finetune - Hard-Label

***

* 2:00:00

***

#### Goldilock Zone

* Soft signals
* Temperature scales softmax

$$P_i = \frac{e^{x_i}}{\sum e^{x_i}} \rightarrow \frac{e^{x_i/T}}{\sum e^{x_i/T}}$$, where $T=\{1,2,3,\ldots\}$.

* Increasing the value of T flattens the graphs, on the other way, reducing the value of T, makes it spiky.
* At $T=1$, entropy is very low.
* But maximum entropy means uniform distribution.

<p align="center">
  <img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/Lecture-10/goldilocks.png" width="500" height="400" />
</p>

***

* 2:10:00

* [Detecting and preventing distillation attacks](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)



***

