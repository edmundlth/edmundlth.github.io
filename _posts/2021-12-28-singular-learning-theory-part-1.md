---
title: Singular Learning Theory - Part 1
tags: [singular-learning-theory, algebraic-geometry, machine-learning]
categories: [singular-learning-theory-lecture-series]
description: 
date: 2021-12-28
math: true
image: 
  src: /2021-12-28/algebraic-geometry-and-statistical-learning-theory-cover-image.jpg
  width: 500
#   height: 50
  alt: image not found
---

This is intended as the first of a series of articles going through core texts for Singular Learning Theory (henceforth SLT) - "Algebraic Geometry and Statistical Learning Theory"[^1] and "Mathematical Theory of Bayesian Statistics"[^2] both written by Sumio Watanabe. This is perhaps mainly my attempt at digesting the text<span sidenote> not for the first time </span> disguised as a series of lectures. 

The target audience I have in mind is anyone familiar with undergraduate level mathematics and is interested in theoretical / mathematical aspect of statistical learning and AI. That being said, the nature of the subject is such that it draws upon tools and concepts from a wide array of mathematical disciplines, spanning algebra to analysis. Some, like probability and statistics, are crucial in the sense that they are the objects of study. Some, like manifold theory, are only required to make sure that the mathematical objects we manipulate are well defined and cover a sufficient generality for the theory to be useful. Others, like algebraic geometry and Schwartz distribution theory, exports crucial theorems that we shall use to prove and understand the central results of SLT. Yet others, like statistical mechanics, are topics where we might find unexpected connections and possible cross-pollination. We shall introduce these topics in their own time when they come up naturally when we explore SLT. Our modest aim regarding these prerequisites is to understand them with sufficient depth to understand the proofs of various results in SLT and to at least understand their significance <span sidenote> like why they are needed and what happen when we can't borrow from them. Of course each of them are profound fields of study in their own right, and if time and energy permit, we shall delve beyond strictly necessary to see the wonder they contain.</span> 

- [What do we aspire to study?](#what-do-we-aspire-to-study)
- [Statistical Background](#statistical-background)
  - [The task of learning](#the-task-of-learning)
  - [Model-Truth-Prior](#model-truth-prior)
  - [Some Important Statistical Estimation Methods](#some-important-statistical-estimation-methods)
  - [Kullback-Leibler Divergence](#kullback-leibler-divergence)
  - [Theory of Statistical Learning](#theory-of-statistical-learning)
  - [Fisher Information Matrix](#fisher-information-matrix)
- [Singular Models](#singular-models)
- [Model Selection](#model-selection)
- [References](#references)

# What do we aspire to study?
Before making definitions and carving out our domain, let's list a few informal ways we think and talk about our field of study <span sidenote> If nothing else, it will help with triangulating what we aspire to study, where the definitions and tools we shall develop are helpful and where there are not.  </span>.
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

## The task of learning
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


## Model-Truth-Prior
A useful and general way of constructing and studying the approximation $\hat{p}$ to a given _truth_ $q(x)$ is to first looking at a large family of distributions, have a way of measuring how well each of them approximates $q$, then come up with a procedure to find the best performing approximation. 

<div class=def>
  <p>
    A statistical <span def> model </span><span sidenote>we might sometimes call this the space of hypothesis</span> is a family of probability distribution $p(x \mid w)$ parameterised by a subset of $d$-dimensional <span def> parameter space </span> $W \subset \R^d$ for some $d \in \N$. We assume that $p(x \mid w)$ at all $w \in W$ have the same support as the true distribution $q(x)$, 
    $$
    \overline{\set{x \in \R^n \wh p(x | w) > 0}} = \overline{\set{x \in \R^n \wh q(x) > 0}}.
    $$
    Given a set of i.i.d. training samples $D_N = \set{X_1, \dots, X_N}$ as before, we call probability of observing $D_N$ for a particular parameter 
    $$
    L_N(w) := \P(X_1, \dots, X_N \mid w) = \prod_{i = 1}^N p(X_i \mid w) 
    $$
    the <span def> likelihood function</span>.  
  </p>

  <p>
    Taking a Bayesian perspective, we shall endow the parameter space $W$ with a <span def> prior probability distribution </span> with density $\varphi(w)$ so that, applying Bayes Theorem, we get the posterior distribution on $W$

    $$
    p(w \mid D_N) = \frac{L_N(w) \varphi(w)}{Z_N} 
    $$

    where the normalising constant in the denominator 

    $$
    Z_N := \int_W L_N(w) \varphi(w) dw = \int_W \prod_{i = 1}^N p(X_i \mid w) \varphi(w) dw
    $$ 

    is an important quantity known as the <span def> model evidence</span>. 
  </p>
</div>


<details open>
  <summary> (Quite) a few things to say about $Z_N$ </summary>
  <p>
    This quantity comes with a few different names<span sidenote>testament to its importance and ubiquity in mathematical sciences.</span>. In Bayesian statistics, it goes by the names "model evidence" or "marginal likelihood". In statistical mechanics, it goes by "partition function" or, when taken a log-transform, the "free energy", denoted as

    $$
    F_N = -\log Z_N
    $$
  </p>

  <p>
    To justify its name, we observe that $Z_N$ depends on the set of random variables $D_N$, hence it is itself a random variable. In fact, it is precisely the probability of observing the data $\set{X_1, \dots, X_N}$ given the model and prior.<span sidenote>
    Perhaps it is clearer in the basic form of Bayes Theorem $P(H|X) = P(X|H) P(H) / P(X)$ thinking of $H$ as a particular hypothesis in a class of hypotheses and $X$ as the observed data. The denominator $P(X) = \sum_{h \in \text{all hypothesis}} P(X|h)P(h)$ is the probability of observing $X$, $P(X)$, given the hypothesis class.
    </span> It is there for the likelihood of the given model-prior pair given the data, hence "model evidence". It is computed by marginalise out the model parameter $w$, hence "marginal likelihood". 
  </p>

  <p>
    We shall also consider a generalisation of the posterior distribution and model evidence where introduce a continuous parameter $\beta > 0$ which we shall interpret as "inverse temperature" $\beta = \frac{1}{T}$:
    $$
    \begin{align*}
    p(w \mid D_N) &= \frac{\varphi(w) \prod_{i = 1}^N p(X_i \mid w)^\beta}{Z_N(\beta)}\\
    Z_N(\beta) &= \int_W \varphi(w) \prod_{i = 1}^N p(X_i \mid w)^\beta dw.
    \end{align*}
    $$

    Note that 
      <ul>
        <li>$\beta = 1$ reduces down to the case of usual Bayes posterior distribution and evidence. </li>

        <li>As $\beta \to \infty$, the posterior distribution converges the delta distribution concentrated at the maximum likelihood estimate of $w$, $\delta(w - \hat{w}_{ML})$, if it exist and is unique in the support of $\varphi(w)$. We can think of this case as taking the temperature down to zero and having the system considers only the ground state specified by the observed data. </li>

        <li>As $\beta \to zero$, the posterior collapsed to the prior. This is the case where we have infinite temperature and observed data has no effect on the system exploring all allowed state at the frequency specified by the prior. </li>
      </ul>
  </p>

  <p>
    To make the analogy to statistical mechanics even tighter, we can introduce the analog to the potential field by considering quantities normalised by the true probabilities of observed data $q(X_i)$.
    $$
    \begin{align*}
    Z^0_N(\beta) &= \frac{Z_N(\beta)}{\prod_{i = 1}^N q(X_i)^\beta} \\
    F^0_N(\beta) &= -\log Z^0_N(\beta) = F_N(\beta) - N\beta S_N \\
    S_N &= -\frac{1}{N} \sum_{i = 1}^N \log(q(X_i))
    \end{align*}
    $$

    where $S_N$ is the empirical entropy. "Empirical" because it is an empirical estimate of the entropy of the true distribution given by 
    
    $$
      S(q) = - \int q(x) \log q(x) dx = \E[-\log q(x)] = \E[S_N].
    $$
  </p>
  <!---TODO: Further post to clarify the following.
  - Discuss link to statistical mechanics. 
  - Discuss the role of inverse temperature.
  - Provide the "exponential forms" of the the above quantities to make precise the connection to stat mech
  --->
</details>


## Some Important Statistical Estimation Methods
With the above objects defined, we can now specify a few ways to construct $\hat{p}$. 

First, there is the [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) method that constructs an approximation to the truth $q$ by finding, among the model parameter space $W$, the parameter $w \in W$ that maximises the likelihood function. The assumption is that, since the data set $D_N$ is generated from $q(x)$, a good approximation of $q$ is the model $p(x \mid w)$ that have the highest likelihood to have generated the same data. 

<div class=def>
  <p>
    The <span def> Maximum Likelihood Estimator (ML) </span> is given by the model parameter that maximises the likelihood function.

    $$
    \hat{w}_{ML} = \mathrm{argmax}_{w \in W} L_N(w)
    $$

    It turns out that it is easier in practice for numerical reasons and more convenient for theoretical discussions to introduce the negative log-likelihood function $l_N(w) = -log L_N$ and the MLE is given by minimising $l_N(w)$, i.e. $\hat{w}_{MLE} = \mathrm{argmin}_{w \in W} l_N(w)$. With the MLE, the approximate distribution is given by the model at $\hat{w}_{MLE}$,

    $$
    \hat{p}_{ML}(x) = p(x \mid \hat{w}_{ML}).
    $$
  </p>
</div>

MLE is a common _point estimation_ method. Meaning an objective function, the likelihood function in this case, is constructed and then an optimisation procedure is employed to find the optimum which gives the desired estimate. It is often assumed that the model space is parametrised in a such a way that the likelihood function has desirable functional properties suitable for optimisation such as differentiability, convexity or, like in the case of large neural networks, convenient form to compute gradient for [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). 

Bringing some Bayesian perspective into the picture, we can construct another point estimate by biasing the likelihood function with prior belief of what parameters $w \in W$ are _a priori_ more likely to occur without having observe any data. This gives us the [Maximum a Posteriori](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) method.

<div class=def>
  <p>
    The <span def> Maximum a Posteriori (MAP) </span> estimate is obtained by maximising the posterior distribution

    $$
    \hat{w}_{MAP} = \mathrm{argmax}_{w \in W} p(w \mid D_N) = \mathrm{argmin}_{w \in W} l_N(w) - \log \varphi(w)
    $$

    where $\varphi(w)$ is the prior distribution. The approximation to $q(x)$ is then given by 

    $$
    \hat{p}_{MAP}(x) = p(x \mid \hat{w}_{MAP}).
    $$
  </p>
</div>

In extreme cases, if we believe<span sidenote>on sound theoretical grounds or just sheer human pig-headedness</span> that the model parameters lies in a certain region and nowhere else, then even if the MLE method claim that the data is most likely to be generated by the model at $\hat{w}_{MLE}$, the MAP estimate might completely disregard that possibility.

Instead of considering one "best" parameter in the model, a fully Bayesian perspective will advocate for taking distributions at all parameters $w \in W$ into account, with the importance of each $p(x \mid w)$ weighted by their posterior probability. 

<div class=def>
  <p>
    The <span def> Bayes predictive distribution </span> for the approximation of $q(x)$ is given by
    $$
    \hat{p}_{B} = p(x | D_N) = \E_{w}\sqbrac{p(x \mid w)} = \int_{W} p(x \mid w) p(w \mid D_N) dw 
    $$
    where $\E_w$ denote taking expectation with respect to the posterior. 
  </p>
</div>

Another method of constructing statistical estimation $\hat{p}$ perhaps rare in practice but holds theoretical importance is that of Gibbs estimation. With Gibbs estimation, every time<span sidenote>I am a little unsure about whether we draw $\hat{w}$ once and for all or perform multiple draws as described.</span> one wish to estimate the probability of $X = x$, one starts by randomly drawing a parameter $\hat{w}$ from the posterior distribution $p(w \mid D_N)$ and answer with $p(X = x \mid \hat{w})$. 
<div class=def>
  <p>
    The <span def> Gibbs estimation</span> method is given by randomly drawing a parameter from the posterior distribution
    $$
    \hat{w}_{gibbs} ~ p(w \mid D_N)
    $$

    and give the approximation to $q(x)$ by 

    $$
    \hat{p}_{gibbs} = p(x \mid \hat{w}_{gibbs}).
    $$
  </p>
</div>


## Kullback-Leibler Divergence
<!--
- How do we measure the goodness of approximation
- D_KL as excess surprise. 
- Some characteristic properties of KL
- Asymptotic Gaussian in regular models and what does a statistical learning theory do.
-->
Next come the problem of how to measure the success or failure of our approximation. We need a quantity that measures the difference between two density functions. Enter the [Kullback-Leibler](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) [divergence](https://en.wikipedia.org/wiki/Divergence_(statistics)). 

<div class=def>
  The <span def> Kullback-Leibler Divergence </span> or KL-divergence between two probability density functions $q(x)$ and $p(x)$ on an open set $A \subset \R^n$ is given by 
  $$
  K(q \mid\mid p) := \int_A q(x) \log \frac{q(x)}{p(x)} dx. 
  $$
</div>

<details>
  <summary> KL-divergence on general probability measure </summary>
  <p>
    <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">KL-divergence</a> can be defined in much greater generality between any two probability measures $P$ and $Q$ over the same measure space $(X, \sigma)$ by 
    $$
    K(Q \mid\mid P) := \int_X \log \frac{dQ}{dP} dQ
    $$
    whenever the <a href="https://en.wikipedia.org/wiki/Radon%E2%80%93Nikodym_theorem#Radon%E2%80%93Nikodym_derivative">Radon-Nikodymn derivative</a> $\frac{dQ}{dP}$ exists 
    <span sidenote>i.e. $Q$ is absolutely continuous with respect to $P$.</span>.

    If there is another measure $\mu$ on $X$ for which $p = \frac{dP}{d\mu}$ and $q = \frac{dQ}{d\mu}$ exists, then by the chain rule, we recover a more familiar form
    $$
    K(Q \mid\mid P) = \int_X \log\brac{\frac{\frac{dQ}{d\mu}}{\frac{dP}{d\mu}}} \frac{dQ}{d\mu} d\mu = \int_X q \log \frac{q}{p} d\mu.
    $$
    In the definition we gave above, we have been using the standard Lebesgue measure on $\R^n$. As another example, if we have the counting measure $\mu_c$ on a discrete measure space $X$ and probability measures $P$ and $Q$ on $X$, then the expression becomes, 
    $$
    K(Q \mid\mid P) = \sum_{x \in X}Q(x) \log \frac{Q(x)}{P(x)}.
    $$
  </p>
</details>

I find it helpful to understand a little about [Shanon's information entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))<span sidenote>and we shall need the concept of information entropy in our study anyway.</span> in order to get an intutitive sense of KL-divergence. Informally, Shanon's information quantifies how surprised one should be when an event that was thought to have probability $p \in [0, 1]$ occurs. Intuitively, the observation "the sun rose in the east this morning" has less information content than "it was raining this morning in Perth" or "the Dow Jones index rose by 3% this morning"<span sidenote>think about which "insider knowledge" - knowledge that dispels prior uncertainty by increasing the event probability to 1 - will you pay more to obtain to give an edge in a bet.</span>. A little more formally, Shanon information is another way of encoding probability with the following properties
 - Information content is a monotonically decreasing non-negative function $I: [0, 1] \to \R_{\geq 0}$ of the probability $p$ of the event.
 - Events with complete certainty have zero information. 
 - Information of independent events adds. 
These properties together with some smoothness assumptions on $I$ forces 

$$
 I(p) = -\log p.
$$



## Theory of Statistical Learning
$$
 -\frac{1}{N} \log L_N(w) \to K(w) - S(q)
$$ 

But minimisation of $-\log L_N(w)$ is not equivalent to minimisation of $K(w)$ (which would give the closest estimate to $q(x)$). So MLE is not the only valid or the best statistical estimation method. 


## Fisher Information Matrix



# Singular Models
Various equivalent definitions of singular models (via FIM and K(w)). Some implications.

# Model Selection 






# References
[^1]: Sumio Watanabe. _Algebraic Geometry and Statistical Learning Theory_. Cambridge University Press, August 2009.

[^2]: Sumio Watanabe. _Mathematical Theory of Bayesian Statistics_. CRC Press, Chapman and Hall/CRC. 2018. 

[^3]: Shannon, C. E. 1948. “A Mathematical Theory of Communication.” The Bell System Technical Journal 27 (3): 379–423.