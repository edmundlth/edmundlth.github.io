---
title: Singular Learning Theory -- Part 1
tags: [singular-learning-theory, algebraic-geometry, machine-learning]
categories: [singular-learning-theory-lecture-series]
description: 
date: 2021-12-28
math: true
image: 
  src: http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/resolution.jpg
  width: 500
#   height: 50
  alt: image not found
---

This is intended as the first of a series of articles going through core texts for Singular Learning Theory (henceforth SLT) - "Algebraic Geometry and Statistical Learning Theory"[^1] and "Mathematical Theory of Bayesian Statistics"[^2] both written by Sumio Watanabe. This is perhaps mainly my attempt at digesting the text<span sidenote> not for the first time </span> disguised as a series of lectures. 

The target audience I have in mind is anyone familiar with undergraduate level mathematics and is interested in theoretical / mathematical aspect of statistical learning and AI. That being said, the nature of the subject is such that it draws upon tools and concepts from a wide array of mathematical disciplines, spanning algebra to analysis. Some, like probability and statistics, are crucial in the sense that they are the objects of study. Some, like manifold theory, are only required to make sure that the mathematical objects we manipulate are well defined and cover a sufficient generality for the theory to be useful. Others, like algebraic geometry and Schwartz distribution theory, exports crucial theorems that we shall use to prove and understand the central results of SLT. Yet others, like statistical mechanics, are topics where we might find unexpected connections and possible cross-pollination. We shall introduce these topics in their own time when they come up naturally when we explore SLT. Our modest aim regarding these prerequisites is to understand them with sufficient depth to understand the proofs of various results in SLT and to at least understand their significance <span sidenote> like why they are needed and what happen when we can't borrow from them. Of course each of them are profound fields of study in their own right, and if time and energy permit, we shall delve beyond strictly necessary to see the wonder they contain.</span> 

# What do we aspire to study?
Let us now set up the statistical objects that shall be our main concern for a long while. Before making definitions and carving out our own domain, let's list a few informal ways we think and talk about our field of study <span sidenote> If nothing else, it will help with triangulating what we aspire to study, where the definitions and tools we shall develop are helpful and where there are not.  </span>.
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
 
 <li> <a href="https://deepmind.com/research/case-studies/alphago-the-story-so-far"> AlphaGo</a> is a machine intelligence that were able to learn from millions of generated examples of games of Go, and learn the pattern of winning strategies so well that it was able to defeat the reigning world champion. </li>
 
 <li> Brains of human and other intelligent species seems to implement complicated and as yet poorly understood system that learn the pattern of their surroundings, able to triage tasks and resources and make decisions critical to the survival of its species. </li>
</ul>
</details>

# Statistical Background
Let's come back down to earth with some definitions. 
In typical statistical learning scenario, we are given a set of examples coming from a data generating process. 

<div class=def>
We are given a <span def> data set </span>

$$
D_N = \set{X_1, \dots, X_N}
$$

of $N \in \N$ <span def> training samples </span> consisting of independently and identically distributed random variables $X_i$ <span> i.e. i.i.d. samples</span> drawn from what we shall call the <span def> true distribution</span> specified by a probability density $q(x)$.
</div>

Unless otherwise specified, the random variables take values in a fixed Euclidean space $\R^n$. For clarity, we shall often have rather strong regularity assumptions on any probability distributions that occurs throughout the development of SLT. Not only do we assume that densities exist for probability distributions we study, we shall also assume at least continuity and likely analyticity<span sidenote> full set of conditions will be specified later on. This setting allow us to leverage heavy duty machineries from algebraic geometry to explicate salient features of learning machines before persuing further generality. Though this seems rather restrictive, it does cover a large class of distributions we encounter in practice like normal mixtures, neural networks, Bayesian networks and much more. </span>

The truth $q$ is what a learning machine wants to access, but can only see the shadow of via the data set $D_N$. For our purpose, we shall formulate the task of a statistical learning machine as one that approximates the true distribution $q$ from the given set of training samples $D_N$.

<div class=def>
A <span def> statistical inference </span> or a <span def> learning machine </span> is a measurable function taking any data set $D_N$ to probability distribution<span sidenote> again, we shall work with distribution on $\R^n$.</span> specified by $\hat{p}(x)$, 

$$
\begin{align*}
\mathrm{LearningMachine}: \set{\text{data set of any size $N$}} &\to \set{\text{probability distributions}} \\
 D_N &\mapsto \hat{p}(x).
\end{align*} 
$$
</div>

Among the concerns of a statistical learning theory includes studying 
 - how to construct such a learning machine
 - how to obtain $\hat{p}$ efficiently<span sidenote> efficiency in terms of computation resources and size of data set.</span> 
 - and how well does $\hat{p}$ thus constructed approximates $q$.  

A useful and general way of constructing and studying $\hat{p}$ is to first looking at a large family of distributions, have a way of measuring how well each of them approximates $q$, then come up with a procedure to find the best performing approximation. 

<div class=def>
A statistical <span def> model </span> is a family of probability distribution $p(x \mid w)$ parameterised by a subset of $d$-dimensional <span def> parameter space </span> $W \subset \R^d$ for some $d \in \N$. 

<br/>
Given a set of i.i.d. training samples $D_N = \set{X_1, \dots, X_N}$ as before, we call probability of observing $D_N$ for a particular parameter 

$$
L_N(w) := \P(X_1, \dots, X_N \mid w) = \prod_{i = 1}^N p(X_i \mid w) 
$$

the <span def> likelihood function</span>.  

<br/>
Taking a Bayesian perspective, we shall endow the parameter space $W$ with a <span def> prior probability distribution </span> with density $\varphi(w)$ so that, applying Bayes Theorem, we get the posterior distribution on $W$

$$
p(w \mid D_N) = \frac{L_N(w) \varphi(w)}{Z_N} 
$$

where the normalising constant in the denominator 

$$
Z_N := \int_W L_N(w) \varphi(w) dw = \int_W \prod_{i = 1}^N p(X_i \mid w) \varphi(w) dw
$$ 

is an important quantity known as <span def> marginal likelihood </span> or <span def> model evidence </span>. 
</div>

With the above objects defined, we can now specify a few ways to construct $\hat{p}$

<div class=def>
The <span def> Maximum Likelihood (ML) </span> method constructs $\hat{p}$ by maximising the likelihood 

$$
\begin{align*}
\hat{w}_{ML} &= \mathrm{argmax}_{w \in W} L_N(w) \\
\hat{p}_{ML}(x) &= p(x \mid \hat{w}_{ML}).
\end{align*}
$$

Bringing some Bayesian perspective into the picture, we have the <span def> Maximum a Posteriori (MAP) </span> method, obtained by maximising the posterior distribution instead

$$
\begin{align*}
\hat{w}_{MAP} &= \mathrm{argmax}_{w \in W} p(w \mid D_N) \\
\hat{p}_{MAP}(x) &= p(x \mid \hat{w}_{MAP}).
\end{align*}
$$

Instead of considering one "best" parameter in the model, a fully Bayesian perspective will advocate for taking distributions at all parameters $w \in W$ into account, with the importance of each $p(x \mid w)$ weighted by their posterior probability. This gives the <span def> Bayes predictive distribution </span> for the approximation of $q(x)$ 
$$
\hat{p}_{B} = p(x | D_N) = \E_{w}\sqbrac{p(x \mid w)} = \int_{W} p(x \mid w) p(w \mid D_N) dw 
$$
where $\E_w$ denote taking expectation with respect to the posterior. 

</div>


<details>
<summary> Comments on $Z_N$ </summary>
</details>


# References
[^1]: Sumio Watanabe. _Algebraic Geometry and Statistical Learning Theory_. Cambridge University Press, August 2009.

[^2]: Sumio Watanabe. _Mathematical Theory of Bayesian Statistics_. CRC Press, Chapman and Hall/CRC. 2018. 