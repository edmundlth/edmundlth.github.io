---
title: Hopfield Networks and Statistical Mechanics
tags: [neural-computation, hopfield-networks, statistical-mechanics, artificial-intelligence]
categories: [machine-learning]
description: 
date: 2021-11-30
math: true
image: 
  src: https://upload.wikimedia.org/wikipedia/commons/4/49/Energy_landscape.png #/2021-11-30/
  width: 800
#   height: 50
  alt: image not found
---

# Introduction



# Model specification

Each neuron<span class=sidenote>perhaps a more neutral word like "node" is better here. But it is undeniable that biological neurons were the inspiration for this model despite various approximations, significant or otherwise, being made.</span> shall receives binary signals from all neurons connected to it and shall itself produces a binary output signal. We shall represent the binary signal output by the $i^{th}$ neuron as $S_i \in \set{-1, +1}$. This output is computed as the sign <span class=sidenote> $\sgn(x) = +1$ if $x \geq 0$ and $-1$ otherwise.</span> a weighted sums of signals $S_j$ of neurons connected to it. Therefore, 
![sign-function](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Signum_function.svg/2560px-Signum_function.svg.png){: .right width="250"}

$$
S_i := \sgn\brac{\sum_j w_{ij}S_j - b_i}
$$

where $w_{ij}$ are the _weights_ or strength of connection from neuron $j$ to $i$ and $b_i$ is the _bias_ or threshold the weighted sums needed to cross before $S_i$ turns on. 

