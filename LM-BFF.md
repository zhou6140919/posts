---
title: LM-BFF
author: Zhou Tong
date: 2022-12-07 17:26:36
tags: [nlp, papers]
categories: nlp
feature: true
mathjax: true
---
# Making Pre-trained Language Models Better Few-shot Learners

<!-- more -->

See [original paper](https://arxiv.org/abs/2012.15723) for more details.

## Introduction

**Motivation**

Inspired by GPT-3, the authors study few-shot learning in a more practical scenario,
where the authors use smaller language models for which fine-tuning is computationally efficient.

**Contributions**

The authors present LM-BFF (better few-shot fine-tuning of language models) a suite of simple
and complementary techniques for finetuning language models on a small number of annotated examples.

The authors' approach includes:
1. **prompt-based fine-tuning together with a novel pipeline for automating prompt generation;**
2. **a refined strategy for dynamically and selectively incorporating demonstrations into each context.**

The authors' experiments demonstrate that our methods combine to dramatically outperform standard fine-tuning
procedures in this low resource setting, achieving up to 30% absolute improvement, and 11% on average across all tasks.
The authors' approach makes minimal assumptions on task resources and domain expertise, and hence constitutes a strong task-agnostic method for few-shot learning.

## Problem Setup

**Task formulation**
 - A pre-trained language model $\mathcal{L} =$ RoBERTa-large
 - Task $\mathcal{D}$
 - Number of training examples per class (2 classes for regression: above or below the median value): $K = 16$
 - Total number of examples: $K\_{tot} = K\times  \left |  \mathcal{Y}  \right |$
 - Training set: $\mathcal{D}\_{train}$
 - Dev set: $\mathcal{D} _{dev}$
 > $\left | \mathcal{D} _{train} \right |  = \left | \mathcal{D} _{dev} \right | $
 > This distinction is important: using a larger development set confers a significant advantage, and subvert their initial goal of learning from limited data.

**Evaluation datasets**
 - GLUE, SNLI
 - SST-5, MR, CR, MPQA, Subj, TREC

1. single-sentence tasks
    The goal is sto make a prediction based on an input sentence $x\_{in} = x\_{1}$, such as whether a movie review is positive or not.
2. sentence-pair tasks
    The goal is to take a pair of input sentences $x\_{in} = (x\_{1}, x\_{2})$ and predcit the relationship between them.

**Evaluation protocol**
> Fune-tuning on small datasets can suffer from instability, and results may change dramatically given a new split of data.

The authors measure average performance across 5 different randomly sampled $\mathcal{D}\_{train}$ and $\mathcal{D}\_{dev}$ splits. They argue that sampling multiple splits gives a more robust measure of performance, and a better estimate of the variance. They sweep multiple hyperparameters for each data sample and take the best setting as measured on the $\mathcal{D}\_{dev}$ of that sample.

## Prompt-based Fine-tuning

Given a masked language model $\mathcal{L}$, the authors first convert input $x\_{in}$ to a token sequence $\tilde{x}$ to a sequence of hidden vectors $\mathbf{h} _k ∈ \mathbb{R} ^d$.

During standard fine-tuning, they take $\tilde{x}\_{single} = [CLS]x\_{1}[SEP]$ or $\tilde{x}\_{pair} = [CLS]x\_{1}[SEP]x\_{2}[SEP]$.

Both in classification and regression tasks, the number of new parameters can be substantial.
> For exampmle, a simple binary classification task will introduce 2,048 new parameters for a RoBERTa-large model -- making it challenging to learn from a small amount of annotated data (e.g., 32 examples).

To solve this probelm, the authors propose *prompt-based fine-tuning*, in which $\mathcal{L}$ is directly taskeed with "auto-completing" natural language prompts.

> If input $x\_{1}$ is "*No reason to watch it.*"
> then the prompt will be like:
> $x\_{prompt} = [CLS]x\_{1}$ It was $[MASK] . [SEP]$
> and let $\mathcal{L}$ decide whether it is more appropriate to fill in "*great*" (positive) or "*terrible*" (negative) for $[MASK]$.

### Classification

Let $\mathcal{M: Y\to  V } $ be a mapping from the task label space to individual words in the vocabulary $\mathcal{V}$ of $\mathcal{L}$.

Then for each xin, let the manipulation $x\_{prompt} = \mathcal{T}(x\_{in})$ be a *masked language modeling* (MLM) input which contains one $[MASK]$ token.

The authors treat their task as an MLM, and model the probability of predicting class $y \in \mathcal{Y}$ as:
$$
p(y|x\_{in}) = p([MASK]=\mathcal{M}(y)|x\_{prompt}) =  \frac{exp(\mathbf{w}\_{\mathcal{M}(y)}\cdot \mathbf{h}\_{[MASK]})}{ {\textstyle \sum_{y'\in \mathcal{Y} }^{} exp(\mathbf{w}\_{\mathcal{M}(y')}\cdot \mathbf{h}\_{[MASK]})} }
$$

where $\mathbf{h}\_{[MASK]}$ is the hidden vector of $[MASK] and $\mathbf{w}\_{v}$ denotes the pre-softmax vector corresponding to $v \in \mathcal{V}$. $\mathcal{L}$ minimizes the cross-entropy loss.
>This approach re-uses the pre-trained weights $\mathbf{w}\_{v}$ and does not introduce any new parameters.
> It also reduces the gap between pre-training and fine-tuning, making it more effective in few-shot scenarios.

### Regression

The authors assume the same basic setup as in classification, but treat the label space $\mathcal{Y}$ as a bounded interval $[v_l, v_u]$. They model the problem as an iterpolation between two opposing poles, {$y_l, y_u$}, with values $v_l$ and $v_u$ respectively.
> For instance, they formulate sentiment analysis task as a regression problem in the range $[0, 1]$, where they slide between “terrible” $(v_l = 0)$ and “great” $(v_u = 1)$.
> In this way, $y$ can be expressed as a *mixture model*:
> $$
> y = v_l \cdot p(y_l | x\_{in}) + v_u \cdot p(y_u | x\_{in})
> $$
> where $p(y_u | x\_{in})$ is the probability of $y_u$, and $p(y_l | x\_{in}) = 1 - p(y_u | x\_{in})$.
> They define $\mathcal{M}: y_l, y_u \to \mathcal{V}$, and model $p(y_u | x\_{in})$ the same as the classification case.
> Then fine-tune $\mathcal{L}$ to minimize the KL-divergence between the inferred $p(y_u | x\_{in})$ and the observed mixture weight, $(y-v_l)/(v_u-v_l)$.

### Manual prompts: the good and the bad
This table summarizes manual templates and label words chosen from each datasest in the experiments. These templates and label words were designed by intuition, and by considering formats used in previous literature.

![prompt-design](/images/prompt-design.png)

According to the pilot study on SST-2 and SNLI. This table below shows that different prompts can lead to substantial differences in final accuracy. **When a template is fixed, the better the label words match the "semantic classes", the better the final accuracy is** (*great/terrible* > *good/bad* > *cat/dog*).
<img src="/images/prompt-differences.png" alt="prompt-differences" width="300" style="display: block; margin: 0 auto"/>

This clearly underlines the importance of selecting good templates and label words.

## Automatic Prompt Generation

 - search for label words
 - search for templates

Assume a classification task, but the process for regression is analogous.

### Automatic selection of label words

**The goal is to construct a label word mapping $\mathcal{M}$ that maximizes the accuracy on $\mathcal{D}\_{dev}$ after fine-tuning.**

Given a fixed template $\mathcal{T}$, searching all possible assignments is
- generally intractable
- prone to overfitting.

> As a simple solution, for each class $c \in \mathcal{Y}$, the authors construct a pruned set $\mathcal{V}^c \subset \mathcal{V}$ of the top $k$ vocabulary words based on their conditional likelihood using the initial $\mathcal{L}$.
> That is, let $\mathcal{D}\_{train}^c \subset \mathcal{D}\_{train}$ be the subset of all examples of class $c$.
> <img src="/images/vc-formula.png" alt="vc" width="400" style="display: block; margin: 0 auto"/>
> where $P\_\mathcal{L}$ denotes the output probability distribution of $\mathcal{L}$.
> To further narrow down the search space, they find the top $n$ assignments over the pruned space that maximize zero-shot accuracy on $\mathcal{D}\_{train}$ (both $n$ and $k$ are hyperparameters).
> Then fine-tune all top $n$ assignments and rerank to find the best one using $\mathcal{D}\_{dev}$.
> They use simpler search process (brute-force).

### Automatic generation of templates

The goal is to generate a diverse set of templates {$\mathcal{T}$} automatically from a fixed set of label words $\mathcal{M}(\mathcal{Y})$.

The authors propose to use T5 to fill in missing spans in its input.

Given an input example $(x\_{in}, y) \in \mathcal{D}\_{train}$, they consider the following simple conversions, denoted as $\mathcal{T}\_{g}(x\_{in}, y)$, for formulating the T5 model inputs:

<center>

$<S_{1}> \to <\mathrm{X}>\mathcal{M}(y)<\mathrm{Y}><S_{1}>$,

$<S_{1}> \to <S_{1}><\mathrm{X}>\mathcal{M}(y)<\mathrm{Y}>$,

$<S_{1}>, <S_{2}> \to <S_{1}><\mathrm{X}>\mathcal{M}(y)<\mathrm{Y}><S_{2}>$.
</center>

<img src="/images/template-generation.png" alt="tg" width="400" style="display: block; margin: 0 auto"/>

They use **beam search** to decode multiple template candidates. Concretely, they use a wide beam width (e.g., 100) to cheaply obtain a large set of diverse templates. Then fine-tune each generated template on $\mathcal{D}\_{train}$ and select the best one using $\mathcal{D}\_{dev}$.

## Fine-tuning with Demostrations

### Training examples as demonstrations

GPT-3's in-context learning is suboptimal as
 - the number of available demonstrations is bounded by the model's maximum input length;
 - mixing numerous random examples from different classes together creates extremely long contexts which can be hard to leverage, especially for small models.

The authors randomly sample one example from each class, convert it into a template and then concatenate them with $x\_{in}$.

![](/images/prompt-based-fine-tuning.png)

$$
\mathcal{T}\left(x_{\text {in }}\right) \oplus \tilde{\mathcal{T}}\left(x_{\text {in }}^{(1)}, y^{(1)}\right) \oplus \cdots \oplus \tilde{\mathcal{T}}\left(x_{\text {in }}^{(|\mathcal{Y}|)}, y^{(|\mathcal{Y}|)}\right)
$$

> At testing time, they still sample demonstration sets from $\mathcal{D}\_{train}$ and ensemble predictions across all sets.

### Sampling similar demonstrations

> If the set of contrastive demonstrations $x\_{in}^{(c)}$ are all dramatically different from each other or from the query $x\_{in}$, then it becomes challenging for the language model to decipher meaningful patterns.

The authors only sample examples that are semantically close to $x\_{in}$. Specifically, they use [SBERT](https://arxiv.org/abs/1908.10084) model to obtain embeddings for all input sentences (concatenation for two sentences). For each $x\_{in}$, they sample from top $r=50\\% $ instances by their similarity score $cos(\mathbf{e}(x\_{in}), \mathbf{e}(x))$.

## Experiments

### Single-prompt results

![](/images/single-prompt-results.png)

1. Prompt-based zero-shot prediction achieves much better performance than the majority class, showing the pre-encoded knowledge in RoBERTa.
2. In-context learning does not always improve over zero-shot prediction, likely because smaller language models are not expressive enough to use off-the-shelf like GPT-3.
3. Prompt-based fine-tuning can greatly outperform standard fine-tuning, both when using a manual prompt or a generated one.
4. Generally, the automatically searched templates can achieve comparable or even higher results than manual ones.
5. Using demonstrations in context leads to consistent gains in a majority of tasks. Fine-tuning with automatically searched templates and sampled demonstration sets achieves a $30\\%$ gain on SNLI compared to standard fin-tuning, and $11\\%$ gain on average.

### Ensemble results

<img src="/images/ensemble-prompt-results.png" alt="epr" width="400" style="display: block; margin: 0 auto"/>

As the results show, an ensemble with multiple templates always improves performance. An ensemble of the same number of automatic templates achieves comparable or better performance than the ensemble of [PET](https://arxiv.org/abs/2001.07676)'s manual prompts. Increasing the number of automatic templates brings further gains.

### Analysis of generated prompts

<img src="/images/analysis-generated-prompts.png" alt="agp" width="400" style="display: block; margin: 0 auto"/>

For automatic prompts, the authors compare template search (Auto T), label word search (Auto L), and a joint variant (Auto T + L) start from manual label words, apply Auto T, and then Auto L. In most cases, Auto T achieves comparable or higher performance than manual ones, and is consistently the **best** variant.

### Analysis of demonstration sampling

<img src="/images/analysis-demonstration-sampling.png" alt="ads" width="400" style="display: block; margin: 0 auto"/>

The figure above shows the importance of sampling similar examples for incorporating demonstrations in context.

### Sample efficiency

<img src="/images/sample-efficiency.png" alt="ads" width="400" style="display: block; margin: 0 auto"/>

This figure illustrates how standard fine-tuning and our LM-BFF compare as K increases. For a simple task such as SST-2 (also see MR, CR and MPQA), despite using only 32 total examples, LM-BFF has already nearly saturated its performance and is comparable to standard fine-tuning over the entire dataset. On the harder task of SNLI, LM-BFF continues to improve as $K$ increases while still maintaining a performance gap over standard finetuning, until the two converge around $K = 256$.

## Discussion

**Limitations of LM-BFF**
 - Overall, the performance still substantially lags behind fine-tuning with thousands of examples, especially for harder tasks.
 - Just like standard finetuning, the results also suffer from high variance.
 - It is practically challenging to expand the search space because of the reliance on some manual design -- either manual templates (for label word search) or manual label words (for template search).
 - LM-BFF favors certain tasks which (1) can be naturally posed as a “fill-in-the-blank” problem; (2) have relatively short input sequences; and (3) do not contain many output classes. 
 > Issues (2) and (3) might be ameliorated with longer-context language models (e.g., [Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)).
 > For tasks that are not straightforward to formulate in prompting, such as structured prediction, **issue (1) is more fundamental**.
 