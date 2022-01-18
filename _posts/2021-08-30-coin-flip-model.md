---
title: Toy Model - Coin Flip
tags: [singular-learning-theory, RLCT, machine-learning, statistics]
categories: [algebraic-geometry, machine-learning]
description: 
date: 2021-08-30
math: true
image: 
  src: /2021-08-30/
  width: 300
#   height: 50
  alt: image not found
---
Following a long tradition of first example in statistical pedagogy, we
will investigate coin flips: a single Bernoulli random variable $\{H, T\} \ni x \sim Bernoulli(w)$ parametrised by a single variable $w \in [0, 1] = W \subset \R$. In this case, we have 

$$
\begin{aligned}
    p(x| w) &= \begin{cases}
        w & x = H\\
        1 - w & x = T
    \end{cases}\\
    q(x) &= p(x| w_0)\\    
    K(w) &= w_0 \log w_0 + (1 - w_0) \log (1 - w_0) - w_0 \log w - (1 - w_0) \log (1 - w)\\
    K_N(w) &= \hat{w}_{MLE} \log \frac{w_0}{w} + (1 - \hat{w}_{MLE}) \log \frac{1 - w_0}{1 - w}, \quad \hat{w}_{MLE} = \frac{\# H}{N} \\
    I(w) &= \frac{1}{w( 1 - w)} > 0 \quad \forall w \in (0, 1)
\end{aligned}    
$$

where the last expression is the (positive definite) Fisher information matrix<span sidenote>admittedly not much of a matrix in 1-D.</span>. This is clearly a regular model. Assuming uniform prior $\varphi = 1$, the Laplace integral can be computed exactly as

$$
\begin{aligned}
    L(n) 
    &= \int_0^1 \exp\left(-nK(w)\right)dw \\
    &= \left(w_0^{w_0}(1 - w_0)^{1 - w_0}\right)^{-n} \int_0^1 w^{n w_0}(1 - w)^{n(1 - w_0)}dw\\
    &= C(w_0)^{-n}B(n w_0 + 1, n(1 - w_0) + 1)
\end{aligned}
$$

where $B(x,y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x + y)}$ is the beta function<span sidenote>even in this extremely simple case, we still need special functions to express quantities of interest.</span>. If the truth is that the coin is fair<span sidenote>$$w_0 = 1/2$$ and $$C(w_0) = 2$$ in this case.</span>, using Stirling's approximation for $$\Gamma$$, we have 

$$
\begin{aligned}
    L(n) 
    &\sim \frac{1}{8\sqrt{\pi}} \left(\frac{n}{2} + 1\right)^{-\frac{1}{2}}
\end{aligned}
$$

which tell us that the RLCT $$= 1/2$$ as expected for a regular model. The posterior and Bayesian predictive distribution is given by<span sidenote>the form of the posterior shows that Beta and Bernoulli distributions forms a conjugate pair.</span>

$$
\begin{aligned}
    p(w | D_N) &= \frac{w^{\# H} ( 1 - w)^{N - \# H}}{B(\# H + 1, N - \# H + 1)} \sim \text{Beta distribution}\\
    p(x| D_N) &= \frac{1}{B(\# H + 1, N - \# H + 1)}\begin{cases}
        B(\# H + 2, N - \# H + 1) & x = H\\
        B(\# H + 1, N - \# H + 2) & x = T. 
    \end{cases}
\end{aligned}
$$ 

These can be simplified into binomial coefficients. Finding the analytic continuation for the zeta function<span sidenote>I haven't been able to do it. The integral is singular near the integration terminals. This makes the evaluation using resolution of singularity seems magical to me.</span>

$$
\begin{aligned}
    \zeta(z) = \int_0^1 \left(-\log 2 - \frac{1}{2}\log w(1 - w)\right)^z dw
\end{aligned}
$$

is not trivial even for the $$w_0 = 1/2$$ case. Though this analysis generalised to any finite number of discrete variables<span sidenote>Beta distribution becomes Dirichlet distribution, binomial becomes multinomial etc\...</span>, analytic expression of quantities of interest are both hard to write down and hard to compute<span sidenote>not to mention issues with numerical stability...</span>. I suppose a lesson here is that Bayesian
statistics is HARD.



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
