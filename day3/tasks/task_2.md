# Task

You are given a dataset containing, for each observation index $\(k = 1,\ldots,K\)$ and category index \(j = 1,\ldots,C\), the values \(\{(s_{jk}, m_{jk})\}\). Consider the likelihood contribution for observation \(k\) defined by

```math
f_k(\theta)
=
\int_{\mathbb{R}}
\prod_{j=1}^{C}
\binom{m_{jk}}{s_{jk}}
\, p_j(y;\theta)^{s_{jk}}
\left(1 - p_j(y;\theta)\right)^{m_{jk}-s_{jk}}
\phi(y)\, dy,
```

where \(\phi(y)\) is a known density (given), and \(p_j(y;\theta)\) is a known model function parameterized by \(\theta\).

## 1. Form the full likelihood over the dataset

$$
L(\theta) = \prod_{k=1}^{K} f_k(\theta),
\qquad
\ell(\theta) = \log L(\theta) = \sum_{k=1}^{K} \log f_k(\theta).
$$

## 2. Compute the maximum likelihood estimate (MLE)

$$
\hat{\theta} = \arg\max_{\theta}\, \ell(\theta).
$$

## 3. Important: Numerical optimization requirement

You must obtain \(\hat{\theta}\) using **numerical optimization** (e.g., gradient-based methods or derivative-free methods). Your implementation must:

- numerically evaluate the integral defining \(f_k(\theta)\),
- maximize \(\ell(\theta)\) (equivalently, minimize \(-\ell(\theta)\)) using an optimization routine,
- report the optimized parameter value \(\hat{\theta}\) and the achieved log-likelihood \(\ell(\hat{\theta})\).

## 4. Method description

Provide a brief description of the optimization method you used (algorithm, initialization, stopping criteria) and any constraints on \(\theta\) (if applicable).