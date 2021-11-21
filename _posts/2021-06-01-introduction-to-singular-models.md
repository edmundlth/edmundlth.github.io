---
title: Introduction to Singular Learning Theory
tags: [singular-learning-theory, algebraic-geometry, machine-learning]
categories: [singular-learning-theory]
description: 
date: 2021-06-01
math: true
---

# Introduction
Let's first set the context. Imagine we are given a data generating process $q(x)$ where we can ask for $N \in \N$ samples[^1], $D_N = \{ X_1, \dots, X_N \}$. Our goal is to *learn* a distribution $p(x)$ from which we can make inferences about the data generating process itself. 

Some examples:

-   **Deterministic data:** 
    If $q$ generates the result of "$1 + 1$", we can set $p(x_1) = 1$ where $x_1 = 2$ is the first "measurement" or request we made to the data generating process. Here the learning process recover everything we wish to know about $q$ just from the first data point, i.e. $p = q$. As such, there is no reason to deviate from this learning process. 

-   **Deterministic with measurement error**
    If $q$ generates the results of ballot count by humans, the above learning process would still be reasonable, but we should perhaps account for human error. We could, for instance, ask for lots of recount set $p(\text{most frequently occuring count}) = 1$. Or perhaps a deterministic result doesn't sit well with us when we know that error can occur, we can set $p(x) =$ proportion of recount that
    turns out to be $x$.

-   **Experiments in empirical science**
    If $q$ generates experimental measurements of physical quantities $(x, y)$ that is governed by some law of nature $y = f_\alpha(x)$ that depends on some parameter $\alpha$ and experimental measurements is marred by (normally distributed) random error, then we have $Y_i - f_\alpha(X_i) \sim N(0, \sigma)$. The value of $\alpha$ can be estimated given a learnt model $p$.

-   **Generalised Linear Models**

-   **AI agents**

-   etc [^2]

In general, we instantiate a large space of *hypothesis*,

$$\Delta = \left\{p = p(x|w) \, \, : \,w \in W\right\}$$

parametrised by $w \in W \subset \R^d$ equipped with a prior $\varphi(w)$ and cast the learning process as an *optimisation procedure* that finds the best hypothesis that explains the observed samples. One way to define "best" is to select $p$ that minimises the *Kullback-Leibler divergence* between $q$ and $p$, i.e.choose $p(x) = p(x \st \hat{w})$ such that $\hat{w}$ minimises 

$$
\begin{aligned}
    K(w) = \E_q\left[\log \frac{q(x)}{p(x| w)}\right] = \int_X q(x) \log \frac{q(x)}{p(x| w)}dx
\end{aligned}
$$

We will investigate the properties of learning machine of this form. Properties of a learning machine that we might care about:

-   Error rate. Generalisation. Generalisation gap:
    $$B_g, G_g, B_t, G_t$$.

-   Data efficiency. Compute efficiency. Behaviour in overparametrised
    regime. Scaling laws. Double descent.

-   Training behaviour. Stochastic noise.



# Real Log Canonical Threshold
For a given statistical model $(p(x \mid w), q(x), \varphi(w))$, the following are equivalent definitions for its real log canonical threshold (RLCT), $\lambda$ and its order $\theta$.

1.  **Largest pole of zeta function of $$K$$**\
    Define the zeta function of $K$ as[^3]:
    $$
    \begin{aligned}
        & \zeta: \C \to \C
        &\zeta(z) = \int_W K(w)^z \varphi(w)dw. 
    \end{aligned}
    $$
    
    The RLCT $\lambda$ is the largest pole of $\zeta$ and $\theta$ the order of the pole at $\lambda$.

2.  **Convergence rate of Laplace integral of $K$**\
    $(\lambda, \theta)$ governs the asymptotic behaviour as $n \to \infty$ of the Laplace integral[^4]: 
    
    $$
    \begin{aligned}
        \int_W \exp\left(-nK(w)\right)\varphi(w) dw \stackrel{n \to \infty} \sim Cn^{-\lambda}\left(\log n\right)^{\theta -1}
    \end{aligned}
    $$ 
    
    for some positive real constant $C$.

3.  **Convergence rate of free energy**\
    Taking the negative logarithm of the previous asymptotic expression gives[^5] 
    
    $$
    \begin{aligned}
        \log \int_W \exp\left(-nK(w)\right)\varphi(w) dw \stackrel{n \to \infty} \sim \lambda \log n - \left(\theta -1\right) \log \log n + O(1). 
    \end{aligned}
    $$

4.  **Asymptotic expansion of density of states near $W_0$**\
    The density of state 
    
    $$
    \begin{aligned}
        v(t) = \int_W \delta\left(t - K(w)\right) \varphi(w) dw
    \end{aligned}
    $$ 
    
    has asymptotic expansion as $t \to 0$

    $$
    \begin{aligned}
        v(t) \sim C t^{\lambda -1} (- \log(t))^{\theta -1} 
    \end{aligned}
    $$ 
    
    for some positive real constant $C$.[^6]

5.  **Volume codimension $W_0$** 

    $$
    \begin{aligned}
        \lambda = \lim_{t \to 0^+} \log_a \frac{V(at)}{V(t)}
    \end{aligned}
    $$ 
    
    where $1 \neq a > 0$ and

    $$V: \R_{\geq 0} \to \R_{\geq 0}$$

    is the volume measure of neighbourhoods of $W_0$ 
    
    $$
    \begin{aligned}
        V(t) = \int_{K(w) < t} \varphi(w) dw. 
    \end{aligned}
    $$

6.  **From resolution of singularity**\
    Hironaka's resolution of singularity for the real analytic function $K(w)$ gives us a proper birational map[^7] $g: U \to W$ such that in the neighbourhood of $w_0 \in W_0$, the zero set
    of $K$ 
    
    $$
    \begin{aligned}
        K(g(u) - w_0) &= u^{2k} = u_1^{2k_1}u_2^{2k_2} \dots u_d^{2k_d}\\
        g'(u) &= b(u)u^h = b(u)u_1^{h_1}u_2^{h_2} \dots u_d^{h_d}
    \end{aligned}
    $$ 
    
    for some $u, k \in \N^d$ and analytic $b(u) \neq 0$. We then have 
    
    $$
    \begin{aligned}
        \lambda = \inf_{w \in W_0} \min_{1 \leq j \leq d}\frac{h_j + 1}{2k_j}
    \end{aligned}
    $$ 
    
    and $\theta$ is given by the number of times the above minimum is achieved.[^8]

7.  **RLCT of ideals of analytic functions**\
    TODO: there are various square roots involved in this that I don't
    really understand


# RLCT for Regular Models
The RLCT of a regular realisable model is given by 

$$\lambda = \frac{d}{2} \quad \theta = 1.$$

We shall use the Laplace integral characterisation of RLCT. We want to
show that 

$$Z^0_n = \int_W \exp\left(-nK(w)\right) \varphi(w) d{w} \sim C n^{-\frac{d}{2}}$$

for some positive constant $$C$$ as $$n \to \infty$$. Since the model is realisable and identifiable, it has unique minimum at $$w^* \in \mathrm{supp}(\varphi)$$. Taylor expansion of $$K$$ centered around $$w^*$$ up to order 2 gives 

$$
\begin{aligned}
        K(w) 
        &= K(w^*) + \nabla K(w^*) \cdot (w - w^*) + \frac{1}{2} (w - w^*)^T \nabla^2K(w^*)(w - w^*)  + O(\left|\, w - w^* \,\right|^3) 
\end{aligned}
$$ 

where $\nabla^2K(w^*)$ is the Hessian of $K$ at $w^*$. That $w^*$ realises the true model and is a local minimum gives us $K(w^*) = 0$ and $\nabla K(w^*) =0$, reducing the above to 

$$
\begin{aligned}
        K(w) &= \frac{1}{2} (w - w^*)^T \nabla^2K(w^*)(w - w^*)  + O(\left|\, w - w^* \,\right|^3). 
\end{aligned}
$$

Substituting the above into the integral, we get, in the limit as $$n \to \infty$$ 

$$
\begin{aligned}
        Z^0_n 
        &\sim \int_W \exp\left(-\frac{n}{2} (w - w^*)^T \nabla^2K(w^*)(w - w^*) \right) \varphi(w) d{w} 
\end{aligned}
$$

which we recognise as a Gaussian integral with precision matrix $$n \nabla^2K(w^*)$$ which is positive definite by assumption. Therefore, we conclude that 

$$
\begin{aligned}
        Z^0_n 
        \sim \varphi(w^*)\sqrt{\frac{(2\pi)^d}{\det\left(n \nabla^2K(w^*)\right)}} 
        = \varphi(w^*)\sqrt{\frac{(2\pi)^d}{\det\left(\nabla^2K(w^*)\right)}} n^{-\frac{d}{2}}.
\end{aligned}
$$

We shall use the characterisation that for any positive $$a \neq 1$$, $$\lambda = \lim_{t \to 0^+} \log_a \frac{V(at)}{V(t)}$$ where $$V$$ is the volume function 

$$
\begin{aligned}
        V(t) = \int_{K(w) \leq t} \varphi(w) dw. 
\end{aligned}
$$ 

By regularity assumption, we have that $$w^*$$ is a non-degenerate critical point of $$K$$ and hence by Morse lemma, there is a local chart $$x(w) = (x_i(w))_{i = 1, \dots, d}$$ in a small enough neighbourhood of $$w^*$$ such that $$K(w) = \cancelto{0}{K(w^*)} + \sum_i x_i(w)^2$$. Therefore, for small enough $$t > 0$$, 

$$
\begin{aligned}
        V(t) = \int_{\sum_i x_i^2 \leq t} \varphi(x) dx
\end{aligned}
$$ 

which is proportional to the volume of a $$d$$-dimensional ball with radius $$\sqrt{t}$$, i.e.
$$V(t) \propto t^{d}$$.[^16] Finally, 

$$
\begin{aligned}
        \lambda = \lim_{t \to 0^+} \log_a \frac{(at)^{d/2}}{t^{d/2}} = \frac{d}{2}. 
\end{aligned}
$$



# Footnotes
[^1]: Throughout, we assume that the process generates i.i.d. samples.
    In particular, $$X_i \sim q$$ for all $$i$$, with $$q$$ unchanging as we
    ask for more samples. However, we note that this is a
    simplification: one would imagine that an (artificial) intelligent
    \"student\" would judiciously ask for more informative examples from
    a \"teacher\" process.

[^2]: TODO: more examples. Contrast different inference tasks.

[^3]: $$\zeta$$ analytically continues to a meromorphic function with
    poles on the negative real axis.

[^4]: which is the deterministic version of the (normalised) evidence
    $$Z^0_n = \int_W \exp\left(-nK_N(w)\right)\varphi(w) dw$$.
    Note that the limiting variable $$n$$ is different from the number of
    training samples $$N$$. This is one place where inverse temperature
    $$\beta$$ can come in: set $$n = \beta k$$.

[^5]: the stochastic version translate as
    $$F^0_n = \lambda \log n - (\theta -1) \log \log n +$$ stochastic
    process of constant order.

[^6]: lots of fixing and clarification needed\...Mellin transform's
    involved.

[^7]: obtained via recursive blow up.

[^8]: This deep result shows that
    $$(\lambda, \theta) \in \Q \times \Z$$.

[^9]: admittedly not much of a matrix in 1-D.

[^10]: even in this extremely simple case, we still need special
    functions to express quantities of interest.

[^11]: $$w_0 = 1/2$$ and $$C(w_0) = 2$$ in this case.

[^12]: the form of the posterior shows that Beta and Bernoulli
    distributions forms a conjugate pair.

[^13]: I haven't been able to do it. The integral is singular near the
    integration terminals. This makes the evaluation using resolution of
    singularity seems magical to me.

[^14]: Beta distribution becomes Dirichlet distribution, binomial
    becomes multinomial etc\...

[^15]: not to mention issues with numerical stability\...

[^16]: Volume of $$d$$-dimensional ball with radius $$r$$ is given by
    $$\frac{\pi^{\frac{d}{2}}}{\Gamma(\frac{n}{2} + 1)}r^d$$.
