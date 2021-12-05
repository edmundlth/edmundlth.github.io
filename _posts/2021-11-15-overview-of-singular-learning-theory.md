---
title: Overview of Singular Learning Theory
tags: [singular-learning-theory, real-log-canonical-threshold, statistics, machine-learning]
categories: [singular-learning-theory]
description: 
date: 2021-11-15
math: true
---

Singular Learning Theory (SLT), pioneered by Sumio Watanabe, 
  - acknowledges the presence of singularities in statistical models. 
  - provides a foundation grounded in Algebraic Geometry to work with singularities. 
  - clarifies foundational results in probability and statistics when regularity assumptions are violated. 
  - studies how properties of model singularities and other algebro-geometric properties affect properties of statistical models. 

**Table of Content**
- [The Task of Learning](#the-task-of-learning)
- [The Mathematics of Learning](#the-mathematics-of-learning)
- [Kullback-Leibler Divergence](#kullback-leibler-divergence)
  - [Generalisation Gap](#generalisation-gap)
  - [Model Selection](#model-selection)
- [Singular Learning Theory](#singular-learning-theory)


# The Task of Learning
The task of a statistical learning machine is to discover structures and properties of a data generating process $X$ with probability density $q(x)$ from a set of examples (training data) $D_N = \set{X_1, \dots, X_N}$ identically and independently drawn from the process $X_i \sim q(x)$.

Indeed, in hopes of recovering the **truth** $q$ itself, we define a **statistical model**, which a family of probability distributions $p(x \mid w)$, parametrised by $d$-dimensional real parameters $w \in W \subset \R^d$ endowed with a **prior** probability density $\varphi(w)$. Given such **model-truth-prior triplet**, a learning machine

$$
	D_N \mapsto p(x|\hat{w})
$$

follows specified algorithmic procedure to search for its best guess $\hat{w}$ or a family of guesses based on information it can glean from examples in $D_N$, so that they can be used to approximate $q(x)$.
	
Maximum Likelihood Estimation (MLE) is a well-known example of such a procedure where a single $\hat{w}$ is found by maximising the average log-likelihood function $L_N(w) = \frac{1}{N}\sum_{i} \log p(X_i \mid w)$ and $q$ is approximated by $p(x \mid \hat{w})$. One can also first apply Bayes Theorem to obtain a posterior distribution 

$$
p(w\|D_N) = \frac{e^{NL_N(w)} \varphi(w)}{\int_W e^{NL_N(w)} \varphi(w)dw}
$$

and set $\hat{w}$ to be its maximum, a.k.a. maximum a posteriori (MAP) estimate. Of particular theoretical significance is the Bayesian estimate, where no single $\hat{w}$ is trusted, instead, the truth $q$ is estimated by the posterior average over $w$, i.e. 

$$
	p^*(x) = \int_W p(x \mid w) p(w \mid D_N) dw. 
$$

As a concrete example of contemporary interest and to illustrate the scale at which modern big data and big models, the Generative Pre-trained Transformer 3 (GPT-3) is a deep neural network model with $d \sim 175$ billion parameters, trained using $N >$ 400 billion natural language tokens scraped from the Web. The put emphasis on the algorithmic or "machine" aspect of learning, GPT-3 is trained with a variant of the workhorse of modern machine learning: Stochastic Gradient Descent, an iterative algorithm where a better $\hat{w}$ is proposed by taking a small step in the opposite direction of the gradient $\nabla_w \L(\hat{w}_{\text{old}})$ of a loss function $\L$. The efficient computation of gradient on a models with large $d$ and $N$ with backpropagation is itself a significant algorithmic breakthrough.



# The Mathematics of Learning
Here we sketch several central mathematical objects that arise from the study of statistical learning machines described above and briefly discuss how singularities arise to necessitate the need for methods from algebraic geometry. 
	
# Kullback-Leibler Divergence
	
First, a quantity is needed to represent how well each probability density $p(x\|w)$ in the model approximate the truth $q(x)$. For this, we use the Kullback-Leibler Divergence or relative entropy as established in information theory

$$
	K(w) := D_{KL}(q \|\| p) = \int q(x) \log \frac{q(x)}{p(x\|w)} dx. \label{kl-div}
$$

It is a non-negative quantity $K(w) \geq 0$ for any $w$ and a density $p(x\|\hat{w}) = q(x)$ almost everywhere if and only if $K(\hat{w}) = 0$. Put another way, $K(w) = \E_X[\log q(x) - \log p(x\|w)]$, it is the **generalisation error** as it expresses the expected difference between (log of) the truth $q$ with a selected model $p(x\|w)$. When the expection $\E_X$ is estimated using training data, we get the empirical version of $K$

$$
    K_N(w) := \frac{1}{N} \sum_{i = 1}^N \log \frac{q(X_i)}{p(X_i\|w)}
$$

also known as the training error. Since, $\E_X[K_N(w)] = K(w)$, as the size $N$ of training data increases, $K_N(w)$ converges pointwise almost surely to $K(w)$ by law of large numbers. Observe that, the quantity above can be rewritten as

$$
    K_N(w) = -L_N(w) + \frac{1}{N}\sum_{i = 1}^N \log q(X_i)
$$

where the second term is the empirical version of the entropy $S(q)$ of $q$. Since only the first term has $w$ dependence, maximising the log-likelihood $L_N(w)$ -- as per MLE -- minimises $K_N(w)$. Unfortunately, as Watanabe pointed out in \cite{Watanabe2009-mg}, minimising $K_N(w)$ does not translate minimising $K(w)$ even with $N \to \infty$ since the following limiting operations do not commute

$$
    \E_X[\min_w K_N(w)] \neq \min_w \E_X[K_N(w)] = \min_w K(w). 
$$

Therein lies the tragedy and opportunity of statistical learning theory. The hope for a single best estimation method is dashed but we are promised a rich theory where we shall study the various mode of convergence $K_N \to K$ in function space of differing topology and their correspondence to different learning machines. 
	
## Generalisation Gap
This naturally leads to the study of generalisation gap $K_N - K$. Since this quantity is a measurable function of the data set $D_N$, i.e. a random variable, we asked for uncertainty quantification and also for its rate of decay as $N \to \infty$ which quantifies "data efficiency": how large a data set do we need to collect before we are confident with our theoretical guarantees. 
	
Observe that, for $w$ where $K(w) > 0$, $K_N(w)$ has finite strictly positive variance, $\sigma^2 > 0$, we can invoke central limit theorem to get the following pointwise convergence in law to a normal distribution

$$
    \sqrt{N}\brac{K_N(w) - K(w)} \xrightarrow{d} N(0, \sigma^2). \label{eq:clt-empirical}
$$

Unfortunately, it is not estimates far from $q$ that interest us, nor does pointwise convergence reveals much about the behaviour of the generalisation gap in the neighbourhood of $\set{w \wh K(w) = 0}$. As our main interest is to approximate the truth $q$, behaviour around neighbourhood of $\set{K(w) = 0}$ constitute the main theme of our study. 
	
Here's where we see that the **geometry** of the set of true parameters $W_0 = \set{w \in W \wh K(w) = 0}$ become important. Aside from the case of \emph{regular statistical models} where $W_0$ is a singleton set, $K(w) = 0$ forms an analytic variety in $\R^d$ in the general case of \emph{singular models}. 
	
%TODO: Picture of singularity in K = 0
	
	
## Model Selection
Another question of great theoretical and practical importance is: How do we determine the relative merits of one model over another? That is, with only training samples $D_N$ and no access to the truth $q$, how do we know that a model $p_1(x\|w)$ is better or worse than another $p_2(x\|w)$? This can be quantified using **model evidence**, 

$$
    Z_N = \int_W \prod_{i = 1}^N p(X_i\|w) \varphi(w) dw = \int_W e^{N L_N(w)} \varphi(w) dw
$$

which is marginalises $w$, giving the probability of observing the training samples $X_i$ in $D_N$ under the model $p(x\|w)$. It can be shown that this quantity is controlled by the following integral of Laplace form involving the generalisation error $K(w)$. *If* there is a unique $\hat{w}$ that minimises $K$, $K(\hat{w}) = 0$, and that the Hessian $\nabla^2K(w)$ or equivalently the Fisher information matrix $I(w)_{ij} = - \E_X[\p_{w_i}\p_{w_j}\log p(X\|w)]$ has full rank at $\hat{w}$, we can obtain asymptotic expansion as $n \to \infty$ using Laplace's method

$$
    \int_W e^{-n K(w)} \varphi(w) dw \sim \sqrt{\frac{2\pi}{\det \brac{n\nabla^2K(\hat{w})}}}\varphi(\hat{w}) = O\brac{n^{-\frac{d}{2}}} \label{eq:laplace}
$$

where $d$ is dimension of the parameter space $W \subset \R^d$. That the dominant order is $\sim n^{-d/2}$ give rise to the famous Bayesian Information Criterion (BIC).\\
	
However, here again we are foiled by the possibility that $K(w) = 0$ might not be a singleton set, but an analytic variety with complicated geometry blocking the use of Laplace method for approximating the integral in its neighbourhood. 
	
# Singular Learning Theory
As we observed above, classical results in elementary probability and statistics are frequently frustrated when the statistical model violates the following regularity conditions:

  - map from parameter to density $w \mapsto p(x\|w)$ is one-to-one.
  - the Fisher information matrix $I(w)$ of the model is positive definite for all $w \in W$.

Statistical models that violates the above assumptions are called (strictly) singular models. With singular models, lots of day-to-day statistical intuitions, tools and results no longer apply, for instance
  - Cramer-Rao bound does not hold.
  - Maximum-likelihood estimator is not longer asymptotically Gaussian.
  - BIC and AIC (Akaike Information Criterion) are no longer approximations of what they claim to approximate. 
	
Worse still, in machine learning and many real world applications of statistics, singular models are the norm, not the exception [Watanabe2007]. In light of this, a new foundation for singular statistical model is needed and it hinges on the following remarkable result from Algebraic Geometry.


**Theorem (Hironaka's resolution of singularities)**[Watanabe2009]

Let $f: W \to \R$ be a non-constant real analytic function from a open neighbourhood of $W$ of the origin in $\R^d$ with $f(0) = 0$. Then there exist a **resolution map** $g: U \to W$ such that 

 - $U$ is a $d$-dimensional real analytic manifold. 
 - $g$ is real analytic.
 - $g$ is a proper map.
 - with $W_0 = \set{w \in W \wh f(w) = 0}$ and $U_0 = g^{-1}(W_0)$, $g: U \setminus U_0 \to W \setminus W_0$ is an analytic isomorphism. 
 - if $p \in U_0$, there is a local chart $u = (u_1, \dots, u_d)$ with $p$ as the origin such that $f(g(u))$ and $g'(u)$ are of the form  
    $$
    f(g(u)) = S u_1^{k_1}u_2^{k_2}\dots u_{d}^{k_d} \\
    g'(u) = b(u) u_1^{h_1}u_2^{h_2} \dots u_{d}^{h_d}
    $$
    where $S \in \set{+1, -1}$ is a sign, $b(u) \neq 0$ is real analytic and $k_i, h_i$ are nonnegative integers. 


To illustrate how such a transformation might help with singular models, we note that, in the desingularised coordinates using the resolution map $g: U \to W$ above, we can extend the result to 

$$
    \sqrt{N}\frac{K_N(g(u)) - K(g(u))}{\sqrt{K(g(u))}} \xrightarrow{d} \xi(u)
$$

where $\xi$ is a Gaussian process. And the asymptotic expansion above can now be done for singular models with 

$$
    \log \int_W e^{-n K(w)} \varphi(w) dw \sim \lambda \log n - (m-1) \log \log n + o(\log \log n)
$$

where $\lambda$, instead of $d/2$ in regular models, is a birational invariant of the variety $K(w) = 0$ known as reallog canonical threshold which can be characterised in many ways as summarised below:

**Proposition** [Watanabe2009-mg, lin-phdthesis]

For a given statistical model $(p(x\|\omega), q(x), \varphi(\omega))$, the following are equivalent definitions for its real log canonical threshold (RLCT), $\lambda$ and its order $\theta$. 
- **Largest pole of zeta function of $K$**
    Define the zeta function $\zeta: \C \to \C$ of $K$ as\footnote{$\zeta$ analytically continues to a meromorphic function with poles on the negative real axis.} 

    $$
        \zeta(z) = \int_\Omega K(w)^z \varphi(\omega)d\omega. 
    $$
    
    The RLCT $\lambda$ is the largest pole of $\zeta$ and $\theta$ the order of the pole at $\lambda$.
        
- **Convergence rate of Laplace integral of $K$**
    $(\lambda, \theta)$ governs the asymptotic behaviour as $n \to \infty$ of the Laplace integral\footnote{which is the deterministic version of the (normalised) evidence $Z^0_n = \int_\Omega \exp\brac{-nK_N(\omega)}\varphi(\omega) d\omega$. Note that the limiting variable $n$ is different from the number of training samples $N$. This is one place where inverse temperature $\beta$ can come in: set $n = \beta k$.}: 
    
    $$
        \int_\Omega \exp\brac{-nK(\omega)}\varphi(\omega) d\omega \stackrel{n \to \infty} \sim Cn^{-\lambda}\brac{\log n}^{\theta -1}
    $$

    for some positive real constant $C$. 
        
- **Convergence rate of free energy**
    Taking the negative logarithm of the previous asymptotic expression gives\footnote{the stochastic version translate as $F^0_n = \lambda \log n - (\theta -1) \log \log n + $ stochastic process of constant order. }
    
    $$
        \log \int_\Omega \exp\brac{-nK(\omega)}\varphi(\omega) d\omega \stackrel{n \to \infty} \sim \lambda \log n - \brac{\theta -1} \log \log n + O(1). 
    $$
        
        
- **Asymptotic expansion of density of states near $W_0$**
        The density of state 
    $$
            v(t) = \int_\Omega \delta\brac{t - K(\omega)} \varphi(\omega) d\omega
    $$
        has asymptotic expansion as $t \to 0$ 
    $$
            v(t) \sim C t^{\lambda -1} (- \log(t))^{\theta -1} 
    $$
        for some positive real constant $C$.\footnote{lots of fixing and clarification needed...Mellin transform's involved.}
        
- **Volume codimension $W_0$**
    
    $$
        \lambda = \lim_{t \to 0^+} \log_a \frac{V(at)}{V(t)}
    $$

    where $1 \neq a > 0$ and $V: \R_{\geq 0} \to \R_{\geq 0}$ is the volume measure of neighbourhoods of $W_0$

    $$
        V(t) = \int_{K(w) < t} \varphi(\omega) d\omega. 
    $$ 
        
- **From resolution of singularity**
    Hironaka's resolution of singularity for the real analytic function $K(\omega)$ gives us a proper birational map\footnote{obtained via recursive blow up.} $g: U \to \Omega$ such that in the neighbourhood of $\omega_0 \in W_0$, the zero set of $K$
    
    $$
    \begin{align*}
        K(g(u) - \omega_0) &= u^{2k} = u_1^{2k_1}u_2^{2k_2} \dots u_d^{2k_d}\\
        g'(u) &= b(u)u^h = b(u)u_1^{h_1}u_2^{h_2} \dots u_d^{h_d}
    \end{align*}
    $$

    for some $u, k \in \N^d$ and analytic $b(u) \neq 0$. We then have 

    $$
        \lambda = \inf_{\omega \in W_0} \min_{1 \leq j \leq d}\frac{h_j + 1}{2k_j}
    $$

    and $\theta$ is given by the number of times the above minimum is achieved. <span sidenote>This deep result shows that $(\lambda, \theta) \in \Q \times \Z$. </span> 
