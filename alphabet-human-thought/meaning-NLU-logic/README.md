## Meaning of Sentences, NLU and Logic [[view code]](http://nbviewer.jupyter.org/github/marcotav/natural-language-processing/blob/master/alphabet-human-thought/meaning-NLU-logic/notebooks/meaning-of-sentences.ipynb) 
![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![image title](https://img.shields.io/badge/ntlk-v3.2.5-yellow.svg) 

**The code is available [here](http://nbviewer.jupyter.org/github/marcotav/natural-language-processing/blob/master/alphabet-human-thought/meaning-NLU-logic/notebooks/meaning-of-sentences.ipynb) or by clicking on the [view code] link above.**

*These notes will be heavily based on this [book](https://www.nltk.org/book/).*

It is not difficult to build a grammar formalisms for translation restricted to some specific tasks. In [book](https://www.nltk.org/book/), it is shown that a simple feature-based grammar formalism can translate English questions into SQL queries.

In this notebook, it will be shown that using logic formalisms one can find more generic translation mechanisms. From [book](https://www.nltk.org/book/)

> One advantage of logical formalisms is that they are more abstract and therefore more generic. If we wanted to, once we had our translation into logic, we could then translate it into various other special-purpose languages. In fact, most serious attempts to query databases via natural language have used this methodology.

### Propositional Logic

> Propositional [...] logic deals with propositions and argument flow. Compound propositions are formed by connecting propositions by logical connectives. 

Logical connectives include e.g. the symbols for not, and and or found [here](https://github.com/marcotav/natural-language-processing/blob/master/alphabet-human-thought/meaning-NLU-logic/images/logic.png).

Consider the [example](https://www.nltk.org/book/):

	[Klaus chased Evi] and [Evi ran away]
    
Representing the sentences by **propositional letters** $\phi$ and $\psi$ and using the connective **&** as logical operator this sentence acquires the structure $\phi\, \&\,\psi$, its **logical form**. 

The convention here will be:

```
import nltk
nltk.boolean_ops()
```
with output:

```
negation       	-
conjunction    	&
disjunction    	|
implication    	->
equivalence    	<->
```

Propositional letters are **well-formed formulas** which from now on we will call **formulas**. Therefore is $\phi$ and $\psi$ are formulas any combination using connectives are also formulas. There are several **truth-conditions**involving formulas such as ~$\phi$ is true in s iff $\phi$ is false in s, a so on. 

Applying the method `.parse` from `nltk.LogicParser()` into e.g. the truth condition -(P & Q) we obtain:

```
lp = nltk.sem.logic.LogicParser()
lp.parse('-(P & Q)')
type(lp.parse('-(P & Q)'))
```

### Inference

The inference:

    Mark is taller than John.
    Therefore, John is not taller than Mark.
    
Is correctly represented as:

    [MtJ,MtJ -> - JtM] / - JtM
    
as we must include the a statement clarifying the implication of the expression 'is taller'.

We can test argument validity using `nltk`:

    from nltk import Prover9
    lp = nltk.sem.logic.LogicParser()
    MtJ = lp.parse('MtJ')
    NotJtM = lp.parse('-JtM')
    R = lp.parse('MtJ -> -JtM')
    prover = nltk.sem.logic.Prover9()
    prover.prove(NotJtM, [MtJ, R])
    
The output is `True`.

### First-Order Logic

> First-order logic uses quantified variables over non-logical objects and allows the use of sentences that contain variables, so that rather than propositions such as Socrates is a man one can have expressions in the form "there exists X such that X is Socrates and X is a man" and there exists is a quantifier while X is a variable.This distinguishes it from propositional logic, which does not use quantifiers or relations.


