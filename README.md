## Natural Language Processing

Using Natural Language Processing (NLP) in a nutshell is a technique that:

> Transforms text data and convert it to features that enable us to build models.


## Index

* [neural-language-model-and-spinoza](#neural-language-model-and-spinoza)
* [sentiment analysis](#sentiment analysis)


## neural-language-model-and-spinoza

<p align="center">
<img src="https://github.com/marcotav/natural-language-processing/blob/master/neural-language-model-and-spinoza/images/Spinoza.jpg" width="350"/>   
</p> 

In this project I built a language model for text generation using deep learning techniques. 

Though natural language, in principle, have formal structures and grammar, in practice it is full of ambiguities. Modeling it using examples and modeling is an interesting alternative. The definition of a (statistical) language model is:

> A statistical language model is a probability distribution over sequences of words. Given such a sequence it assigns a probability to the whole sequence.

The use of neural networks has become one of the main approaches to language modeling. Three properties can describe this neural language modeling (NLM) approach succinctly:

> We first associate words in the vocabulary with a distributed word feature vector, then express the joint probability function of word sequences in terms of the feature vectors of these words in the sequence and then learn simultaneously the word feature vector and the parameters of the probability function.

In this project I used Spinoza's Ethics (*Ethica, ordine geometrico demonstrata*) to build a NLM.

## sentiment-analysis

In this project we will perform a kind of "reverse sentiment analysis" on a dataset consisting of movie review from Rotten Tomatoes. The dataset already contains the classification, which can be positive or negative, and the task at hand is to identify which words appear more frequently on reviews from each of the classes. In this project, the Naive Bayes algorithm is used, more specifically the Bernoulli Naive Bayes.
