# Crashcourse: LLMs for Social Scientists

![](images/transformers-crash.png)

This workshop provides an introduction to fundamentals of NLP and LLMs for social scientists. It covers basic and advanced text representation, fundamentals of machine learning, transformer architectures, and applied questions regarding LLMs. The course consists of twelve 90-minute sessions, mostly divided into a lecture with a conceptual focus, and a tutorial covering implementation in python. The course is designed to provide a fast overview of major topics in the application of LLMs. It covers most content rather superficially, aiming to provide students with a good intuition of each concept and code as a starting point to implement their own ideas.

## Day 1

#### Representing Text

### Session 1

**Intro to Python & Text Representation**

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-lecture-1)

🧑‍💻 [Tutorial 1: Intro to Python](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/01a_python.ipynb)

🧑‍💻 [Tutorial 2: Pandas & basic text representation](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/01b_text_reps.ipynb)

#### Further Reading

- **Introduction to text representation for social scientists**: Grimmer, J., Roberts, M. E., & Stewart, B. M. (2022). Bag of Words. In *Text as data: A new framework for machine learning and the social sciences*. Princeton University Press.
- [pandas cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Regular Expressions Cheatsheet](https://regexr.com/)


### Session 2

**Embeddings**

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
- **Scaling Representatives with Document Embeddings**: Rheault, Ludovic, and Christopher Cochrane. 2020. “Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora.” Political Analysis 28 (1): 112–33.
- **Studying over-time Changes in Word Meaning**: Rodman, Emma. 2020. “A Timely Intervention: Tracking the Changing Meanings of Political Concepts with Word Vectors.” Political Analysis 28 (1): 87–111.

## Day 2

#### Machine Learning

### Session 1

**Intro to Supervised Machine Learning**

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-session-3-ml)

🧑‍💻 [Tutorial 1: Intro to Supervised Machine Learning with `scikit-learn`](https://colab.research.google.com/github/nicolaiberk/Imbalanced/blob/master/01_IntroSML_Solution.ipynb)

🧑‍💻 [Tutorial 2: Hackathon](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/03b_hackathon.ipynb)

### Session 2

**Intro to Transformer Models**

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-session-4-transformer/)

🧑‍💻 [Tutorial 1: Contextualized Embeddings, Tokenization, and Inference with Transformers](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/04a_tokens_attention.ipynb)

🧑‍💻 [Tutorial 2: Fine-tuning Transformer Models](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/04b_finetuning_bert.ipynb)

#### Further Reading

- **Simple Explanation of Transformer Architecture**: Tunstall, L., Von Werra, L., & Wolf, T. (2022). Hello Transformers. In: *Natural language processing with transformers*. " O'Reilly Media, Inc.".
- **Original Transformer Paper**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- **Paper introducing BERT Architecture**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) (Devlin et al., NAACL 2019)

**Interactive Tools**

- [Interactive Neural Network Plaground](https://playground.tensorflow.org) by Tensorflow. Play around with network architecture and hyperparameter choices to gain an intuitive understanding of neural networks.

## Day 3

#### Generative Models

### Session 1

**Generative Transformers**

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-session-5-generative/#/title-slide)

🧑‍💻 [Tutorial 1: Annotation with Generative Models](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/05a_prompting.ipynb)

🧑‍💻 [Tutorial 2: API Calls & Structured Output](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/05b_api.ipynb)

#### Further Reading

- [Google prompting guide](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf)

**Visualizations**

- [LLM Visualization by Brendan Bycroft](https://bbycroft.net/llm): Full interactive visualization of GPT Architecture with simple explanations of each step in the architecture.
- [Jay Alamar's Illustrated Transfromer](https://jalammar.github.io/illustrated-transformer/): Accessible visual explanation of transformer architecture.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) Interactive visualization of tranformer forward pass, focusing on attention and impact of specific hyperparameters.


### Session 2

**Using LLMs in Social Science Research**

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-session-6-use-llms/#/title-slide)

<!-- 🧑‍💻 [Tutorial 1: Building a Chatbot for Retrieval-Augmented Generation](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/06a_rag.ipynb) -->

#### Further Reading

- [Debiasing machine-learning estimates](https://naokiegami.com/dsl/articles/intro.html)
