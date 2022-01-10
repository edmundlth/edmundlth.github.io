---
title: Hartshorne Exercises - Chapter 1.1
tags: [hartshorne, algebraic-varieties]
categories: [algebraic-geometry]
date: 2021-11-20
math: true
image: 
  src: /2021-11-20/
  width: 300
#   height: 50
  alt: image not found
---

<details open>
<summary>Exercise 1</summary>
Let $Y$ be a plane curve defined by the equation $y = x^2$ in $A^2_k$. We shall show that its coordinate ring $\A(Y)$ is isomorphic to a one variable polynomial ring $k[t]$. 

<p proof>
First we shall note that the coordinate ring is given by $k[x, y] / I(Y)$ and by Hilbert Nullstellensatz, $I(Y) = \sqrt(y - x^2)$. However, since $k[x, y] \cong k[x][y]$ and $k[x]$ is a UFD<span sidenote> $R$ UFD $\implies R[x]$ UFD.</span>


Hence, our task is to construct an isomorphism $\varphi: A(Y) \cong k[t]$. 

We shall do this by specifying the action of $\varphi$ on the generators $\set{x, y}$ of $k[x, y]$ and check that it descent to a well-defined map on the quotient $A(Y) = k[x, y] / \sqrt{y - x^2}$ as follow

$$
\begin{align*}
x &\mapsto t \\
y &\mapsto t^2.
\end{align*}
$$

This is well defined on $A(Y)$ since 
</p>
</details>