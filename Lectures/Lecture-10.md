## Inference using Smaller Models
1. Finetuning
2. End to end finetuning research project
3. Distillation

***

* How do we take essence of a large model and transfer it to smaller model
* Large Model (70B)
* Small Model (7B)
* [OpenAI Platform](https://platform.openai.com)
* Billing
* API Key Usage

***

* 10:00

#### Reparatization - Why finetuning actually works?
* When we trian a large model (with billion of parameters) its **intrinsic dimension (ID)** reduces
* The intrinsic dimension of a pre-trained large model is way less than that it started out with.
* Whole model can be re-parametrized now in a very small space now.
* The model makes cross-connections during pre-training, which might not have existed before. Same model can now be re-presented in slightly lesser parameter space (called re-parametrization of original model).
* **PCA**: Take a model and represent it in lower dimension!
* Teaching soft signals to smaller model (pointing to a direction)!
* Related to Manifold Learning

***

* RAG: It is like an open book example!
* Finetuning (Studying new material one night before exam!). Re-wiring of brain based on NEW information.
* [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
  * In this paper, we argue that analyzing finetuning through the lens of intrinsic dimension (ID) provides us with empirical and theoretical intuitions to explain this remarkable phenomenon.
  * We empirically show that common pretrained models have a very low intrinsic dimension (ID); in other words, there exists a low-dimension reparameterization that is as effective for finetuning as the full-parameter space.

***

* 25:00
 
* RAG is for knowledge
* Finetuning is for getting the model to learn the patterns (Sign of intelligence).

***
* Llama-3-70B -- Dataset -- Finetune smaller model
* Llama-3-70B -- Customer Support ChatBot --DataSet --Re-wire Llama-7B
* Llama-3-8B often outperforms Llama-3-70B

***

* [The LLM Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook)

***

* 38:00

#### Subliminal Learning (Hidden Signals in Data!)

* [Subliminal Learning: Language models transmit behavioral traits via hidden signals in data](https://arxiv.org/abs/2507.14805)
* [Language models transmit behavioural traits through hidden signals in data](https://www.nature.com/articles/s41586-026-10319-8)

```
* Input (preferences injected through prompts) --> Large Model --> Output: Data
* Generated Data --> Fintunes a Small Model 
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
* **Example**: The student model is trained on Data. Data is generared by a Base Large Model, which has preferences. Theses preference is injected through a Prompt.

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

***
