## Sentence Structure [[view code]](http://nbviewer.jupyter.org/github/marcotav/natural-language-processing/blob/master/alphabet-human-thought/sentence-structure/sentence-structure.ipynb) 
![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![Image title](https://img.shields.io/badge/gensim-0.3.4-blue.svg)


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

### Context-Free Grammar (CFG)

Consider the example:
```
grammar1 = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)
```


A `RecursiveDescentParser` used below is
> A simple top-down CFG parser that parses texts by recursively expanding the fringe of a Tree, 
and matching it against a text.

```
sentence = "the dog saw a man in a park".split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sentence):
    print(tree)
```

The output is:
```
(S
  (NP (Det the) (N dog))
  (VP (V saw) (NP (Det a) (N man) (PP (P in) (NP (Det a) (N park))))))
(S
  (NP (Det the) (N dog))
  (VP (V saw) (NP (Det a) (N man)) (PP (P in) (NP (Det a) (N park)))))

```
T sentence "the dog saw a man in a park" using the grammar above given origin to two trees because of its ambiguous structure, in this case, preposicional phrase attachment ambiguity. In the first tree, the seeing occurred in the park. In this case, NP is "the dog" and the seeing act refers to it.

     [the dog] [saw a man in the park]

In the second, the man was in the park but the dog could be outside, looking at the park. In this case, the first of the right branch is "a man in the park". So the sentence is:

     [the dog saw][a man in a park]
     
## TBC
