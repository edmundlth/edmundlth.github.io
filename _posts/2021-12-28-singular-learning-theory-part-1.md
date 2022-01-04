---
title: Singular Learning Theory -- Part 1
tags: [singular-learning-theory, algebraic-geometry, machine-learning]
categories: [singular-learning-theory-lecture-series]
description: 
date: 2021-12-28
math: true
image: 
  src: /
  width: 300
#   height: 50
  alt: image not found
---

This is intended as the first of a series of articles going through a core text for Singular Learning Theory (henceforth SLT) - "Algebraic Geometry and Statistical Learning Theory" written by Sumio Watanabe[^1]. This is perhaps mainly my attempt at digesting the text<span sidenote> not for the first time </span> disguised as a series of lectures. 

The target audience I have in mind is anyone familiar with undergraduate level mathematics and is interested in theoretical / mathematical aspect of statistical learning and AI. That being said, the nature of the subject is such that it draws upon tools and concepts from a wide array of mathematical disciplines, spanning algebra to analysis. Some, like probability and statistics, are crucial in the sense that they are the objects of study. Some, like manifold theory, are only required to make sure that the mathematical objects we manipulate are well defined and cover a sufficient generality for the theory to be useful. Others, like algebraic geometry and Schwartz distribution theory, exports crucial theorems that we shall use to prove and understand the central results of SLT. Yet others, like statistical mechanics, are topics where we might find unexpected connections and possible cross-pollination. We shall introduce these topics in their own time when they come up naturally when we explore SLT. Our modest aim regarding these prerequisites is to understand them with sufficient depth to understand the proofs of various results in SLT and to at least understand their significance <span sidenote> like why they are needed and what happen when we can't borrow from them. Of course each of them are profound fields of study in their own right, and if time and energy permit, we shall delve beyond strictly necessary to see the wonder they contain.</span> 

# What do we aspire to study?
Let us now set up the statistical objects that shall be our main concern for a long while. Before making definitions and carving out our own domain, let's list a few imprecise ways we think and talk about our field of study <span sidenote> If nothing else, it will help with triangulating what we aspire to study / solve / explain </span>.
 - We are studying systems that can learn from examples. 
 - We are studying algorithms or physical devices that perceive the world and learn its patterns. 
 - We are studying ways to reason and make decision with imperfect information and in the presence of randomness. 
 - We are studying models of systems with randomness and how to find the best approximation for the particular part of the world under observation. 
 - We are studying, or at least trying to study what we think is, intelligence. In particular, machine intelligence, of how machine can learn patterns and make intelligent decisions. 


<details>
<summary> Examples of objects of study that falls under these aspirations. </summary>
<ul>
 <li> A device that implement the calculation $\sum_{i = 1}^6 f(i)/6$ can make decision on whether a given payout $f$ of a game using a six-sided dice is worth playing. </li> 
 
 <li> Perhaps a better, more intelligent, device / machine would actually uses observations on 10000 dice throw and implement $\sum_{i = 1}^6 f(i) \frac{\mathrm{freq}(i)}{10000}$ instead. </li>
 
 <li> A statistician can use a sample of human height measurements $\set{h_1, \dots, h_n}$ and estimate the distribution of heights of comparable human population, or to make a good guess on the probability that a child would grow to be taller than 150cm. </li>
 
 <li> The natural language model GPT-3 is a massive and complicated approximation to human languages. </li>
 
 <li> [AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far) is a machine intelligence that were able to learn from millions of generated examples of games of Go, and learn the pattern of winning strategies so well that it was able to defeat the reigning world champion. </li>
 
 <li> Brains of human and other intelligent species seems to implement complicated and as yet poorly understood system that learn the pattern of their surroundings, able to triage tasks and resources and make decisions critical to the survival of its species. </li>
</ul>
</details>

# Statistical Background
That's more than enough dreaming for now, time for some definitions. 

In typical statistical learning scenario, we are given a set of examples coming from a data generating process. 

<div class=def>
A set of $N \in \N$ samples 
</div>


In this article, we shall
 1. set up the background of statistical learning 
 2. define and discuss a few important statistical objects 
 3. give a brief overview of Singular Learning Theory. 


# What to expect from this series
- what shall we do with other prerequisites. 
- mention some (modest) aspirations? Where I hope this would end up. 
- List of questions that we shall attempt to tackle. 
- brief overview of SLT and main references

# Setting up Statistical Learning Theory


# The Need for Statistical Learning theory - why it is not trivial


# The Need for SLT

# Possible immediate questions...
- Why real analytic? 
- Generality using algebraic geometry -- lower bound problem. 
- Practicality - how many models has been investigated? difficulty of computing quantities in SLT? numerical approximation? 
- Is this just Bayesian theory? 
- How does this relates to Deep Learning? Mention survey of "theory of deep learning". 





- what's next: some simple examples. 


# References
[^1]: Sumio Watanabe. _Algebraic Geometry and Statistical Learning Theory_. Cambridge University Press, August 2009.