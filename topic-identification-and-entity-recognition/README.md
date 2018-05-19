## Topic Identification and Entity Recognition [[view code]](http://nbviewer.jupyter.org/github/marcotav/natural-language-processing/blob/master/topic-identification-and-entity-recognition/topic-identification/notebooks/topic-identification.ipynb) 
![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/ntlk-v3.2.5-yellow.svg) ![Image title](https://img.shields.io/badge/gensim-0.3.4-blue.svg)

**The code is available [here](http://nbviewer.jupyter.org/github/marcotav/natural-language-processing/blob/master/topic-identification-and-entity-recognition/topic-identification/notebooks/topic-identification.ipynb) or by clicking on the [view code] link above.**

<p align="center">
  <a href="#wv"> Word Vectors </a> •
  <a href="#gen"> Gensim dictionary class and corpus </a> •
  <a href="#mct"> Most common terms </a> 
</p>

This project was originally from [here](https://www.datacamp.com/courses/natural-language-processing-fundamentals-in-python).

<a id = 'wv'></a>
### Word Vectors

From [Wikipedia](https://en.wikipedia.org/wiki/Word_embedding):

> Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers.

Word vectors are multi-dimensional representation of word which allows one to obtain relationships between words. These relationships are obtained by NLP algorithms based on how the words are used throughout a text corpus. An example is the difference between word vectors. [The difference is similar](https://www.datacamp.com/courses/natural-language-processing-fundamentals-in-python) between words such as man and women and kind and queen.

<a id = 'gen'></a>
### `Gensim` dictionary class and corpus

This can be best explained with an example. Consider the list `quotes` containing quotes from the Chinese philosopher and writer [Lao Tzu](https://en.wikipedia.org/wiki/Laozi):
- `word_tokenize ` tokenizes the strings `quotes` (after converting tokens to lowercases and dropping stopwords)
- The `Dictionary` class creates a mapping with an id for each token which can be seen using `token2id`. 
- A `Gensim` corpus transforms a document into a bag-of-words using the tokens ids and also their frequency in the document. The corpus is a list of sublists, each sublist corresponding to one document

Since we will be counting tokens I introduced some extra repeated words in the quotes!


```
import nltk
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize

quotes = ['When I let go of what I am am, I become become become what what I might be',
          'Mastering others is strength strength strength. Mastering Mastering Mastering Mastering Mastering\
          yourself yourself yourself yourself is true power',
          'When you are content content content to be simply yourself and do not compare or compete, everybody will respect you',
          'Great acts are made up of small deeds deeds deeds deeds',
          'An ant on the move does more than a dozing dozing dozing dozing ox',
          'Anticipate Anticipate Anticipate Anticipate the difficult by managing the easy',
          'Nature does not hurry, yet everything everything everything everything is accomplished accomplished accomplished',
          'To see things in the seed, that is genius genius genius genius genius genius']

tokenized_quotes = [word_tokenize(quote.lower()) for quote in quotes] 
from stop_words import get_stop_words
en_stop_words = get_stop_words('en')
tokenized_quotes_stopped = []
for token_lst in tokenized_quotes:
    tokenized_quotes_stopped.append([i for i in token_lst if not i in en_stop_words if len(i) > 4])

dictionary = Dictionary(tokenized_quotes_stopped) 
corpus = [dictionary.doc2bow(doc) for doc in tokenized_quotes_stopped]
```
The dictionary is:
```
{'become': 0, 'might': 1, 'mastering': 2, 'others': 3, 'power': 4, 'strength': 5, 'compare': 6, 'compete': 7, 'content': 8, 'everybody': 9, 'respect': 10, 'simply': 11, 'deeds': 12, 'great': 13, 'small': 14, 'dozing': 15, 'anticipate': 16, 'difficult': 17, 'managing': 18, 'accomplished': 19, 'everything': 20, 'hurry': 21, 'nature': 22, 'genius': 23, 'things': 24}
```
and the corpus is:
```
[[(0, 3), (1, 1)], [(2, 6), (3, 1), (4, 1), (5, 3)], [(6, 1), (7, 1), (8, 3), (9, 1), (10, 1), (11, 1)], [(12, 4), (13, 1), (14, 1)], [(15, 4)], [(16, 4), (17, 1), (18, 1)], [(19, 3), (20, 4), (21, 1), (22, 1)], [(23, 6), (24, 1)]] 
```
In the corpus above consider the first and second lists corresponding to the first and second quotes:

    corpus[0] -> [(0, 3), (1, 1)]
    corpus[1] -> [(2, 6), (3, 1), (4, 1), (5, 3)]

These tuples represent:

    (token id from the dictionary, token frequency in the quote)

The third tuple (2, 6) of corpus[1] e.g. says that the token 'mastering' with id = 2 (which can be obtained using `.get( )`) from the dictionary appeared six times in corpus[1]. 

<a id = 'mct'></a>
### Most common terms

To obtain the most common terms in the second quote (and across all quotes) we can proceed as follows. First we sort the tuples in `corpus[1]` by frequency. Note the syntax here. The key defines the sorting criterion which is the function:

    w[x] = element of w with index x
    
This function is implemented using **lambda**. 
```
quote = corpus[1]
quote = sorted(quote, key=lambda w: w[1], reverse=True)
```

Using `dictionary.get(word_id)` we find the words corresponding to the id `word_id` in the dictionary. The `for` below identifies the frequency each term appears in the first quote of the corpus:
```
for word_id, word_count in quote[:3]:
    print(word_id,'|', word_count,'|',dictionary.get(word_id),'|', word_count)
```
We now create an empty dictionary using `defaultdict` to include a total word count:
```
from collections import defaultdict
total_word_count = defaultdict(int)
```
We will also need `itertools.chain.from_iterable` to joins all tuples.. From the docs:

> Make an iterator that returns elements from the first iterable until it is exhausted, then proceeds to the next iterable, until all of the iterables are exhausted. Used for treating consecutive sequences as a single sequence. 

Using a `for` loop we use create a word_id entry in the empty dictionary and for each of the words corresponding to these id we sum all its occurrences in the corpus:

```
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 
for word_id, word_count in sorted_word_count[:5]:
    print('Frequency of the term'+' "'+dictionary.get(word_id)+'"'+' is:', word_count)
```
We end up with the number of times each word appears in the full corpus:

```
Frequency of the term "mastering" is: 6
Frequency of the term "genius" is: 6
Frequency of the term "deeds" is: 4
Frequency of the term "dozing" is: 4
Frequency of the term "anticipate" is: 4

```



