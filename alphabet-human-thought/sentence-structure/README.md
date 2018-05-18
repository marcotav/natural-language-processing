# NLP: Sentence Structure [[view code]](http://nbviewer.jupyter.org/github/marcotav/natural-language-processing/blob/master/alphabet-human-thought/sentence-structure/sentence-structure.ipynb) 
![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/ntlk-v3.2.5-yellow.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg) ![Image title](https://img.shields.io/badge/pandas-0.22.0-red.svg) ![Image title](https://img.shields.io/badge/matplotlib-v2.1.2-orange.svg) ![Image title](https://img.shields.io/badge/gensim-0.3.4-blue.svg)


**The code is available [here](http://nbviewer.jupyter.org/github/marcotav/natural-language-processing/blob/master/alphabet-human-thought/sentence-structure/sentence-structure.ipynb) or by clicking on the [view code] link above.**

These notes will be heavily based on this [book](https://www.nltk.org/book/).

## Goals

The goals of this notebook are:
- Understand how to use a formal grammar to describe the structure of sentences
- Use syntax trees to represent the structure of sentences
- Understand how parser analyze sentences and build a syntax tree

### Framework

Ths framework will be of generative grammar. From Wikipedia:

> Generative grammar is a linguistic theory that regards grammar as a system of rules that generates exactly those combinations of words that form grammatical sentences in a given language.

In other words:
> A language is the collection of all grammatical sentences and a grammar is a formal notation that can be used to generate the members of this set.

As will be explained, grammars use recursive **productions** of the form $S \to S \,\text{and}\, S$.


### Sentence structure and meaning

Following [book](https://www.nltk.org/book/) I will use the following quote my G. Marx to illustrate ambiguities found in language:

*"While hunting in Africa, **I shot an elephant in my pajamas**. How he got into my pajamas, I don't know."* (G. Marx)

To examine the ambiguous phrase sentence in bold we first define a grammar. About the syntax, `nltk.CFG.fromstring` returns the context-free grammar (CFG) corresponding to the input string.

> In formal language theory, a context-free grammar (CFG) is a certain type of formal grammar: a set of production rules that describe all possible strings in a given formal language. Production rules are simple replacements. 

```
import nltk
groucho_grammar = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")
```

The meaning of this symbols can be found [here](https://github.com/marcotav/alphabet-human-thought/blob/master/images/syntactic-categories.png). Using `nltk.ChartParser` we generate:


```
sentence = 'I shot an elephant in my pajamas'.split(' ')
parser = nltk.ChartParser(groucho_grammar)
for tree in parser.parse(sentence):
    print(tree)
```

A graph of the tree structure can be found [here](https://github.com/marcotav/alphabet-human-thought/blob/master/images/tree_cfg.png). The two trees differ with respect to where the shooting occured:
- In the first the elephant was inside the pajamas and the shot is a camera shot.
- In the second the elephant was shot with a gunby a shooter using pajamas.

**We see that by identifying the structure of the sentences, the meaning becomes more easily identifiable.**
