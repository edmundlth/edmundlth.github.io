---
title: Ideals, Varieties and Algorithms - Exercises - Chapter 1
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
  <summary>Exercise 1.2</summary>
  <p> 
    <b> a) Consider the polynomial $g(x, y) = x^2y + y^2 x$ over the field $\F_2$. Show that $g(x, y) = 0$ for all $(x, y) \in \F_2^2$. </b> <br>
    If either $x = 0$ or $y = 0$, both terms vanishes. And $g(1, 1) = 1 + 1 = 0$. 
  </p>

  <p> 
    <b> b) Find a nonzero polynomial in $\F_2[x,y,z]$ which vanishes at every point of $\F_2^3$. Try to find one involving all three variables. </b> <br>
    We shall take inspiration from Boolean logic. We note that the elements $0, 1 \in \F_2$ corresponds to "false" and "true" values and polynomials in $x, y$ corresponds to propositional formulas. More precisely, we can check the following correspondence between logical operations with polynomial expressions:
    <ul>
      <li> $x \wedge y \to xy$ </li> 
      <li> $x \vee y \to x + y + xy$ </li> 
      <li> $\lnot x \to (x + 1)$. </li> 
    </ul>
    A prototypical contradiction looks like $\lnot x \wedge x$ which translate to the polynomial $x(x + 1)$ which we can check is indeed zero as a function on $\F_2^3$. Constructing a polynomial with all three variables is now easy, we can just multiply by the remaining variables: $(x + 1)xyz$. One can check that, for both possible values of $x \in \set{0, 1}$, the first two factor evaluate to zero and hence killing the whole polynomial. 
  </p>

  <p>
    c) Generalising the above to arbitrary number of variables, we get $(x_1 + 1)\prod_{i = 1}^n x_i$ is a non-zero polynomial in $\F_2[x_1, \dots, x_n]$ that is zero as a function on $\F_2^n$. 
  </p>
</details>

<details open>
  <summary> Exercise 1.3 </summary>
  <p> 
    
  </p>

</details>