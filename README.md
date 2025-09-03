# CUSO Workshop: LLMs for Social Scientists

3-5 September 2025, UNIL Lausanne

This workshop provides an introduction to fundamentals of NLP and LLMs for social scientists. It covers basic and advanced text representation, fundamentals of machine learning, transformer architectures, and applied questions regarding LLMs. The course consists of six three-hour sessions, each divided into a lecture with a conceptual focus, and a tutorial covering the implementation in python. The course is designed to provide a fast overview of major topics in the application of LLMs. It covers most content rather superficially, aiming to provide students with a good intuition of each concept and code as a starting point to implement their own ideas.

## Wednesday, Sept. 3rd

### Morning Session (10:30 - 13:30)

#### Intro to Python & Text Representation

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-lecture-1)

🧑‍💻 [Tutorial 1: Intro to Python](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/01a_python.ipynb)

🧑‍💻 [Tutorial 2: Pandas & basic text representation](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/01b_text_reps.ipynb)

**Further reading**

- Grimmer, J., Roberts, M. E., & Stewart, B. M. (2022). Bag of Words. In *Text as data: A new framework for machine learning and the social sciences*. Princeton University Press.
- [pandas cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

### Afternoon Session (14:30 - 17:30)

#### Embeddings

🖥️ [Lecture Slides](https://nicoberk.quarto.pub/llm_ws-lecture-2)

🧑‍💻 [Tutorial 1: Intro to embedding manipulation with `gensim`](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/02a_embeddings.ipynb)

🧑‍💻 [Tutorial 2: Scaling Word Embeddings & Document Embeddings](https://colab.research.google.com/github/nicolaiberk/llm_ws/blob/main/notebooks/02b_embeddings_2.ipynb)

**Explainer**

McCormick, C. (2016, April 19). [Word2Vec tutorial - The skip-gram model](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). *Chris McCormack's Blog*.

**Tools**

- [Regular Expressions Cheatsheet](https://regexr.com/)
- [`gensim` documentation and tutorials on embeddings](https://radimrehurek.com/gensim/auto_examples/index.html#documentation)

**Social Science Applications**

- Kozlowski, Austin C, Matt Taddy, and James A Evans. 2019. “The Geometry of Culture: Analyzing the Meanings of Class Through Word Embeddings.” American Sociological Review 84 (5): 905–49. 
- Kroon, Anne C, Damian Trilling, and Tamara Raats. 2021. “Guilty by Association: Using Word Embeddings to Measure Ethnic Stereotypes in News Coverage.” Journalism & Mass Communication Quarterly 98 (2): 451–77.
-  Rheault, Ludovic, and Christopher Cochrane. 2020. “Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora.” Political Analysis 28 (1): 112–33.
- Rodman, Emma. 2020. “A Timely Intervention: Tracking the Changing Meanings of Political Concepts with Word Vectors.” Political Analysis 28 (1): 87–111.

**Foundational Papers**

- **Word Embeddings**: Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. “Efficient Estimation of Word Representations in Vector Space.” arXiv Preprint arXiv:1301.3781. 
- **Document Embeddings**: Le, Quoc, and Tomas Mikolov. 2014. “Distributed Representations of Sentences and Documents.” In International Conference on Machine Learning, 1188–96. PMLR.
- **Embedding Regression**: Rodriguez, Pedro L, Arthur Spirling, and Brandon M Stewart. 2023. “Embedding Regression: Models for Context-Specific Description and Inference.” American Political Science Review 117 (4): 1255–74. 

## Thursday, Sept. 4th

### Morning Session (9:30 - 12:30)

Intro to Machine Learning

### Afternoon Session (13:30 - 16:30)

Intro to Transformer Models


## Friday, Sept. 5th

### Morning Session (9:30 - 12:30)

Generative Transformers

### Afternoon Session (13:30 - 16:30)

Using LLMs in Social Science Research/tbd

<!-- 

## Day 1

### Introduction & Representing Text

*Lecture I: Introduction & Applications of NLP in the Social Sciences

*Tutorial I: Intro to Python

*Lecture II: Representing Text: Bag-of-Words

*Tutorial II: Bag-of-Words and Scaling

### Embeddings

#### Lecture I: Working with Embeddings

[Slides]()

#### Tutorial I: Intro to Word Embeddings with `gensim`

[Notebook]()

Lecture II: Advanced Embeddings

Tutorial II: Scaling Word Embeddings & Document Embeddings

## Day 2

### Supervised Machine Learning

Lecture I: The basic process of Supervised Machine Learning & Bias-variance tradeoffs

Tutorial I: Supervised ML with `scikit-learn`

Lecture II: Basics of Neural Networks

*Tutorial II: Classification with embeddings, hackathon: best model with and without embeddings

### Introduction to Transformers: The Encoder

Lecture I: Advanced Tokenization & Contextualized Embeddings

Tutorial I: Tokenization, attention, inference with transformers

Lecture II: The Encoder, Training a Transformer

Tutorial II: Fine-tune your own BERT model

## Day 3

### Advanced NLP & the Decoder

*Lecture I: ?? (Decoder architecture, training generative models, hyperparameters)

*Tutorial I: ?? (tracking your experiments/hyperparameter tuning with wandb)

*Lecture II: ?? (Climate Impact of LLMs & PEFT, Bias & Debiasing, ...)

*Tutorial II: ?? (PEFT?)

### RAG & other Shananigans, Q&A

*Lecture I: ?? ()

*Tutorial I: ?? (tracking your experiments/hyperparameter tuning with wandb?)

*Lecture II: ?? ()

*Tutorial II: ?? (Building your own chatbot with RAG?) -->
