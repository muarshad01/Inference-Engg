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

#### Reparatization
* When we trian a large model (with billion of parameters) its intrinsic dimension reduces. *
* The intrinsic dimension of a pre-trained model is way less than that it started out with.
* Whole model can be reparametrized now in a very small space now.
* That's why finetuning actually works.
* The model makes cross-connections during pre-training, which might not have existed before. Same model can now be represented in slightly lesser parameter space (called reparametrization of original model).
* PCA: Take a model and represent it in lower dimension!
* Teaching soft signals to model (pointing to a direction)
* ID (Intrinsic dimensionality)
* Related to Manifold Learning

***

* RAG (It is like an open book example!)
* Finetuning (Studying new material one night before exam!). Rewiring of brain based on new information.
* [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
  * In this paper, we argue that analyzing fine-tuning through the lens of intrinsic dimension provides us with empirical and theoretical intuitions to explain this remarkable phenomenon.
  * We empirically show that common pre-trained models have a very low intrinsic dimension; in other words, there exists a low dimension reparameterization that is
as effective for fine-tuning as the full parameter space

***

* 25:00
 
* RAG is for knowledge
* Finetuning is for getting the model to learn the patterns (Sign of intelligence).


* Llama-3-70B -- Customer Support ChatBot --DataSet --Re-wire Llama 70B
* Llama-3-8B often outperforms Llama-3-70B
* Llama-3-70B -- Dataset -- Finetune smaller model

***

* [The LLM Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook)

***

* 38:00

#### Subliminal Learning (Hidden Signals in Data!)

* [Subliminal Learning: Language models transmit behavioral traits via hidden signals in data](https://arxiv.org/abs/2507.14805)
* [Language models transmit behavioural traits through hidden signals in data](https://www.nature.com/articles/s41586-026-10319-8)
* Anthropic (Sonnet) -- Pharma --finetune --> Drug Model
* **A Certain way of manipulation, which is impossible to detect!!!! How would you detect these soft signals (subliminal learning - hidden signals in data - transfer behavioral traits).**
* **Political consequence also for Nations to enforce their power and preference. What if it is transmited through social engineering and models**

***
