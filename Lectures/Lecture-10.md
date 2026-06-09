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


***
