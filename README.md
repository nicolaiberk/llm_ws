# Using Large Language Models for Frame Measurement and Other Shenanigans

#### DiMES Workshop at FU Berlin, July 9 and 10

![](images/llms_framing.png)

This workshop provides an introduction to fundamentals of natural language processing (NLP) and large language models (LLMs) for political communication research. It covers basic and advanced text representation, fundamentals of machine learning, the inner workings of LLMs, and applied questions regarding LLMs. The course consists of seven 90-minute sessions and one 45-minute session, mostly divided into a lecture with a conceptual focus, and a tutorial covering implementation in python. The course is designed to provide a fast overview of major topics in the application of LLMs. It covers most content rather superficially, aiming to provide students with a good intuition of each concept and code as a starting point to implement their own ideas.

<!-- How can large language models be used to measure framing and other relevant concepts in political communication? This practical workshop introduces you to the essentials of leveraging these powerful models for your research. You will gain a basic understanding of text embeddings, transformer architectures, and the fundamentals of training and fine-tuning models.  We will delve into the practical use of these models using the Huggingface library in Python to effectively use existing models, fine-tune your own models, and deploy them via APIs. One session will delve into frame measurement with modern NLP methodology and how different methods align with different conceptualizations. Finally, we will cover model bias and its mitigation in the use of LLM predictions for statistical inference. -->

## Day 1 (July 9)

#### Representing Text

### Session 1

**Intro to Python & Text Representation**

Intro, Colab/Python, Text Representation (BoW) - *consider VaDER and stance detection discussion, esp Bestvater Munroe paper; ideally foreshadow ML application with simple regression model - maybe start with agenda setting/topics as application*


🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-lecture-1)

🧑‍💻 [Tutorial 1: Intro to Python](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/01a_python.ipynb)

🧑‍💻 [Tutorial 2: Pandas & basic text representation](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/01b_text_reps.ipynb)

#### Further Reading

- **Introduction to text representation for social scientists**: Grimmer, J., Roberts, M. E., & Stewart, B. M. (2022). Bag of Words. In *Text as data: A new framework for machine learning and the social sciences*. Princeton University Press.
- [pandas cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Regular Expressions Cheatsheet](https://regexr.com/)
- [Bestvater SE, Monroe BL. Sentiment is Not Stance: Target-Aware Opinion Classification for Political Text Analysis. Political Analysis. 2023;31(2):235-256. doi:10.1017/pan.2022.10](https://www.cambridge.org/core/journals/political-analysis/article/sentiment-is-not-stance-targetaware-opinion-classification-for-political-text-analysis/743A9DD62DF3F2F448E199BDD1C37C8D)


### Session 2

**Embeddings (the short version)**

*add more substantive application (make sure to discuss a paper using embeddings; maybe for stance detection? or associative framing? - could use approach by Kroon et al and apply it to another dataset?)*

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-lecture-2)

🧑‍💻 [Tutorial 1: Intro to embedding manipulation with `gensim`](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/02a_embeddings.ipynb)

🧑‍💻 [Tutorial 2: Scaling Word Embeddings & Document Embeddings](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/02b_embeddings_2.ipynb)

#### Further Reading

- **Explainer on Algorithm behind Word Embeddings**: McCormick, C. (2016, April 19). [Word2Vec tutorial - The skip-gram model](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). *Chris McCormack's Blog*.
- [`gensim` documentation and tutorials on embeddings](https://radimrehurek.com/gensim/auto_examples/index.html#documentation)

**Foundational Papers**

- **Word Embeddings**: Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. “Efficient Estimation of Word Representations in Vector Space.” arXiv Preprint arXiv:1301.3781. 
- **Document Embeddings**: Le, Quoc, and Tomas Mikolov. 2014. “Distributed Representations of Sentences and Documents.” In International Conference on Machine Learning, 1188–96. PMLR.
- **Embedding Regression**: Rodriguez, Pedro L, Arthur Spirling, and Brandon M Stewart. 2023. “Embedding Regression: Models for Context-Specific Description and Inference.” American Political Science Review 117 (4): 1255–74. 

**Social Science Applications**

- **Studying word Meaning with Embeddings**: Kozlowski, Austin C, Matt Taddy, and James A Evans. 2019. “The Geometry of Culture: Analyzing the Meanings of Class Through Word Embeddings.” American Sociological Review 84 (5): 905–49. 
- **Measuring Bias and Stereotypes with Word Embeddings**: Kroon, Anne C, Damian Trilling, and Tamara Raats. 2021. “Guilty by Association: Using Word Embeddings to Measure Ethnic Stereotypes in News Coverage.” Journalism & Mass Communication Quarterly 98 (2): 451–77.
- **Studying over-time Changes in Word Meaning**: Rodman, Emma. 2020. “A Timely Intervention: Tracking the Changing Meanings of Political Concepts with Word Vectors.” Political Analysis 28 (1): 87–111.
- **Scaling Representatives with Document Embeddings**: Rheault, Ludovic, and Christopher Cochrane. 2020. “Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora.” Political Analysis 28 (1): 112–33.

### Session 3

**Intro to Supervised Machine Learning**

*focus on logic, extend validation, substantive application: partisan differences in speech/?? (Berk)?*

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-lecture-3)

🧑‍💻 [Tutorial 1: Intro to Supervised Machine Learning with `scikit-learn`](https://colab.research.google.com/github/nicolaiberk/Imbalanced/blob/master/01_IntroSML_Solution.ipynb)

🧑‍💻 [Tutorial 2: Hackathon](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/03b_hackathon.ipynb)

#### Further Reading

- [Google's Machine Learning Crach Course](https://developers.google.com/machine-learning/crash-course)
- [scikit-learn documentation](https://scikit-learn.org/stable/). Not only a documentation of the major library for machine learning, but a great resource of tutorials and explainers on 
machine learning.

**Social Science Applications**


[Missing]


### Session 4 (Short session)

*buffer/build your own measure/applications in ComSci & PolSci*

*certainly discuss applications and how different methods might capture different things; if using 'build' your own measure, make sure to have a notebook for each method as starting point and consider removing hackathon*



## Day 2 (July 10)

### Session 1

**A quick & dirty introduction to LLMs & the Huggingface Ecosystem**

*drop the fine-tuning, anything about architecture - focus on inference!*

*substantive application: framing (Berk 2025)*

<!-- Why to keep HF focus but throw out the fine-tuning (might still supply notebook): Open-weight models via HF (Llama, Mistral, your Apertus-70B work) matter for privacy-sensitive data — survey responses, non-public platform data — where sending things to an OpenAI/Anthropic API isn't an option. That's a real methodological argument, not just a technical nicety.

reproducibility as a selling point: a downloaded checkpoint doesn't change under you the way an API-served model does. Good contrast to raise right before the generative LLM section's discussion of reproducibility/versioning. -->

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-lecture-4)

🧑‍💻 [Tutorial 1: Contextualized Embeddings, Tokenization, and Inference with Transformers](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/04a_tokens_attention.ipynb)

<!-- 🧑‍💻 [Tutorial 2: Fine-tuning Transformer Models](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/04b_finetuning_bert.ipynb) -->

#### Further Reading

- [**Huggingface Explainer on Subword Tokenization**](https://huggingface.co/learn/llm-course/en/chapter2/4)
- **Simple Explanation of Attention & Transformer Architecture**: Tunstall, L., Von Werra, L., & Wolf, T. (2022). Hello Transformers. In: *Natural language processing with transformers*. " O'Reilly Media, Inc.".
- **Original Transformer Paper**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- **Paper introducing BERT Architecture**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) (Devlin et al., NAACL 2019)

**Interactive Tools**

- [Interactive Neural Network Playground](https://playground.tensorflow.org) by Tensorflow. Play around with network architecture and hyperparameter choices to gain an intuitive understanding of neural networks.

### Session 2

**Generative Transformers**

*Generative LLMs, zero/few/dynamic/few-shot/RAG/synthetic annotation, chat structure, structured annotation with pydantic*

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-lecture-5)

🧑‍💻 [Tutorial 1: Annotation with Generative Models](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/05a_prompting.ipynb)

🧑‍💻 [Tutorial 2: API Calls & Structured Output](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/05b_api.ipynb)

🧑‍💻 [Tutorial 3: Informed Prompting and Retrieval-Augmented Generation](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/06a_rag.ipynb)

#### Further Reading

- [Google prompting guide](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf)

**Visual Guides**

- [LLM Visualization by Brendan Bycroft](https://bbycroft.net/llm): Full interactive visualization of GPT Architecture with simple explanations of each step in the architecture.
- [Jay Alamar's Illustrated Transfromer](https://jalammar.github.io/illustrated-transformer/): Accessible visual explanation of transformer architecture.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) Interactive visualization of tranformer forward pass, focusing on attention and impact of specific hyperparameters.


### Session 3

**Using LLMs for frame measurement and other shenanigans**

*Application: how to use these tools for measurement in the Social Sciences; Framing definitions and measures (Gruber, Own Paper, Others?); Validation, Reproducibility (local model hosting?) & Debiasing*

#### Further Reading

- [Debiasing machine-learning estimates](https://naokiegami.com/dsl/articles/intro.html)

### Session 4

**Using LLMs in Social Science Research**

*Buffer/Tools and Shenanigans for Research (Claude Code/Agents/Tool Use/Automation)*

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-lecture-6)


#### Further Reading

