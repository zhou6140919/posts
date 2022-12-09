---
title: Towards Unsupervised Language Understanding and Generation by Joint Dual Learning
author: Zhou Tong
date: 2022-12-08 19:25:36
tags: [nlp, papers]
categories: nlp
feature: true
mathjax: true
---

# Towards Unsupervised Language Understanding and Generation by Joint Dual Learning

<!-- more -->

See [original papaer](https://aclanthology.org/2020.acl-main.63) for more details.

## Introduction

**Motivation**

In modular dialogue systems NLU and NLG are two critical components. However, the dual property between understanding and generation has been rarely explored. The prior work still learned both components in a *supervised* manner.

**Contribution**

 - This paper proposes a general learning framework using the duality between NLU and NLG, where *supervised* and *unsupervised* learning can be flexibly incorporated for joint training.
 - This work is the first attempt to exploit the dual relationship between NLU and NLG towards unsupervised learning.
 - The benchmark experiments demonstrate the effectiveness of the proposed framework.

## Proposed Framework

![](/images/dual-framework.png)
The proposed joint dual learning framework, which comprises *Primal Cycle* and *Dual Cycle*. The framework is agnostic to learning objectives.

### Problem Formulation

For both NLU and NLG tasks, there are two spaces:
 - the semantics space $\mathcal{X}$
 - the natural language space $\mathcal{Y}$
> NLG: $\mathcal{X} \rightarrow \mathcal{Y}$
> NLU: $\mathcal{Y} \rightarrow \mathcal{X}$

Given $n$ data pairs sampled from the joint space $\mathcal{X} \times \mathcal{Y}$.
A typical strategy for the optimization problem is based on maximum likelihood estimation (MLE) of the parameterized conditional distribution by the trainable parameters $\theta_{x \rightarrow y}$ and $\theta\_{y \rightarrow x}$ as below:

$$
f(x ; \theta_{x \rightarrow y})=\underset{\theta_{x \rightarrow y}}{\arg \max } P(y \mid x ; \theta_{x \rightarrow y}),
$$
$$
g(y ; \theta_{y \rightarrow x})=\underset{\theta_{y \rightarrow x}}{\arg \max } Pt(x \mid y ; \theta_{y \rightarrow x}).
$$

**Datasets** E2E NLG challenge dataset: 50k instances in the restaurant domain.

> Semantic frame: name[Bibimbap House], food[English], priceRange[moderate], area [riverside], near [Clare Hall]
> Sentence: *Bibimbap House is a moderately priced restaurant who’s main cuisine is English food. You will find this local gem near Clare Hall in the Riverside area.*

### Joint Dual Learning

Previous work: Based on RL or standard supervised learning, and the models are trained *separately*.
**This work: A complete cycle of transforming data from the original space to another space then back to the original space should be exactly the same as the original data, which could be viewed as the ultimate goal of a dual problem.**
$$
g(f(x)) \equiv x
$$

The objective is to achieve the *perfect complete cycle* of data transforming by training two dual models in a *joint* manner.

### Algorithm Description

![](/images/joint-dual-algorithm.png)

1. Primal Cycle transforms the semantic representation $x$ to sentences by function $f$, then computes the loss by the given loss function $l_1$
2. Then predicts the semantic meaning from the generated sentences and computes the loss by the given loss function $l_2$, then fully train the models based on the computed loss.
3. Dual Cycle is symmetrically formulated.

### Learning Objective

NLU task is to predict corresponding slot-value pairs of utterances, which is a multi-label classification problem, the authors utilized the binary cross entropy loss.

NLG uses the cross entropy loss.

The authors introduce the reinforcement learning objective into the framework. In the experiments, policy gradient method is used for optimization, the gradient could be written as:
$$
\nabla \mathbb{E}[r]=\mathbb{E}[r(y) \nabla \log p(y \mid x)]
$$

### Reward Function
 - explicit reward
 - implicit reward

### Explicit Reward

To evaluate the quality of generated sentences, two explicit reard functions are adopted.

**Reconstruction Likelihood** The authors use the reconstruction likelihood at the end of the traiing cycles as a reward function:

\begin{array}{ll}
\log p\left(x \mid f\left(x_{i} ; \theta_{x \rightarrow y}\right) ; \theta_{y \rightarrow x}\right) & \text { Primal, }
\end{array}
\begin{array}{ll}
\log p\left(y \mid g\left(y_{i} ; \theta_{y \rightarrow x}\right) ; \theta_{x \rightarrow y}\right) & \text { Dual. }
\end{array}

**Automatic Evaluation Score** For NLG, BLEU and ROUGE measure n-gram overlaps between the generated outputs and the reference texts. For NLU, F-score is used to indicate the understanding performance.

### Implicit Reward

In addition to explicit signals, a "softer" feedback signal may be informative. Model-based methods estimating data distribution is designed to provide such soft feedback.

**Language Model** The authors use a RNN model to compute the joint probability of the generated sentences, measuring their naturalness and fluency. This language model is learned by a cross entropy objective in an unsupervised manner:
$$
p(y)=\prod_{i}^{L} p\left(y_{i} \mid y_{1}, \ldots, y_{i-1} ; \theta_{y}\right)
$$

**Masked Autoencoder for Distribution Estimation (MADE)** For NLU, the output semantic frame $x$ contains the core concept of a certain setence, furthermore, the slot-value pairs are not independent to others, because they correspond to the same individual utterance.

> By interrupting certain connections between hidden layers, the authors could enforce the variable unit $x_d$ to only depend on any specific set of variables, not necessary on $x_{<d}$; eventually, they could still have the joint distribution by product rule:
> $$
> p(x)=\prod_{d}^{D} p\left(x_{d} \mid S_{d}\right)
> $$
> where $d$ is the index of variable unit, $D$ is the total number of variables, and $S_d$ is a specific set of variable units.
> Because there is no explicit rule specifying the exact dependencies between slot-value pairs in our data, they consider various dependencies by ensembles of multiple decomposition by sampling different sets Sd and averaging the results.

### Flexibility of Learning Scheme

**Straight-Through Estimator** In many tasks the learning targets are discrete. So the operation like argmax does not have any gradient value, forbidding the networks be trained via backpropagation. Therefore it is difficult to directly connect a primal task and a dual task and jointly train these two models due to the above issue.

The Straight-Through (ST) estimator is a widely applied method due to its simplicity and effectiveness.

> The idea of StraightThrough estimator is directly using the gradients of discrete samples as the gradients of the distribution parameters. Because discrete samples could be generated as the output of hard threshold functions or some operations on the continuous distribution
 
<img src="/images/straight-through-estimator.png" width="400" style="display: block; margin: 0 auto"/>

**Distribution as Input** For NLU, the inputs are the word tokens from NLG, so the authors use the predicted distribution over the vocabulary to perform the weighted-sum of word embeddings. For NLG, the model requires semantic frame vectors predicted by NLU as the input condition; in this case, the probability distribution of slot-value pairs predicted by NLU can directly serve as the input vector.

**Hybrid Objective** This framework is agnostic to learning algorithms. FOr example, you could apply supervised learning on NLU in the first half of Primal Cycle and reinforcement learning on NLG to form a hybrid training cycle. Because two models are trained jointly, the objective applied on one model would potentially impact on the behavior of the other.

**Towards Unsupervised Learning** In the algorithm, you can apply only $l_2$ in the Primal Cycle, and only $l_1$ in the Dual Cycle. Such flexibility potentially enables people to train the models based on unpaired data in an unsupervised manner.

## Experiment

Preprocessing: trimming punctuation marks, lemmatization, lowercasing.

Evaluation Metircs: 
    - BLEU, ROUGE (1, 2, L) for NLG
    - F1 measure for NLU

### Model

A gated recurrent unit (GRU) with fully-connected layers at ends of GRU for both NLU and NLG.

 - Batch size: 64
 - Optimizer: Adam
 - Epochs: 10
 - Hidden size: 200
 - Word embedding size: 50

### Results and Analysis

![](/images/results-and-analysis.png)

Explicit feedback like reconstruction and automatic scores is more useful for boosting the NLG performance. However, the implicit feedback is more informative for improving NLU, where MADE captures the salient information for building better NLU models.

### Qualitative Analysis

![](images/qualitative-analysis.png)