---
title: Mengzi
author: Zhou Tong
date: 2021-10-31 21:00:11
tags: [nlp, papers]
categories: nlp
mathjax: true
---

# Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese

<!-- more -->

## Problems

Methods like knowledge distillation and model compression techniques are still not solutions to avoid massive computing resources or training steps.

## Solutions

- Strategies fo well-designed objectives can significantly improve the model capacity without the need to enlarge the model size.
- Mengzi models including discriminative, generative, financial, and multimodal model variants, are capable of a wide range of language and vision tasks.

![Mengzi models](/images/mengzi-models.jpg)

## Setup

- Data Processing: 300GB from Chinese Wikipedia, Chinese News, and Common Crawl
- Architecture: RoBERTa with MLM
- Pre-training Details: LAMB optimizer, FP16, [deepspeed](https://github.com/microsoft/DeepSpeed).

## Experiments

1. Tasks

    Chinese Language Understanding Evaluation (CLUE) benchmark.

    one sentence input: Adding a linear classifier on top of the [CLS] token to predict label proabilities.

    two sentences input: [CLS] Question [SEP] Passage [SEP] plus two linear output layers to predict the probability of each token being the start and end positions of the answer span like BERT.

    POS + NE + MLM + NSP + [SOP](https://arxiv.org/abs/1909.11942)

    QA input: [CLS] Question \|\| Answer [SEP] Passage [SEP], then predict the probability of each answer on the representations from [CLS] token.

2. Implementation

    Optimizer: Adam (learning rate in {8e-6, 1e-5, 2e-5, 3e-5}) with a warm-up rate of 0.1 and L2 weight decay of 0.01.

    The batch size is selected in {16, 24, 32}. The epochs is set in [2, 5] depending on tasks.

    POS and NE tags are annotated by SpaCy.

    Dynamic Gradient Correction: to solve the problem of MLM causing the disturbance of original sentence structure, which leads to the loss of semantics and improve the difficulty off prediction. More details will be provided in latter version.

3. Fine-tuning Strategies
    - **Knowledge Distillation** Calculate Kullback-Leibler divergence of the contextualized hidden states from teacher and student models respectively for the same input sequence, which is minimized during fine-tuning.
    - **Transfer Learning** Leverage the parameters from the trained model on the CMNLI dataset to initialize the model training for related datasets like $C^3$.
    - **Choice Smoothing** For multi-choice or classification tasks, combining different kinds of training objectives would lead to better performance. Combining the cross-entropy loss and binary cross-entropy loss to help the model larn features from different granularity.
    - **Adversarial Traning** To help the model generalize to unseen data, we apply a smoothness-inducing adversarial regularization technique following [Jiang et al](https://arxiv.org/abs/1911.03437). (2020) to encourage the output of the model not to change much when injecting a small perturbation to the input.
    - **Data Augmentation** It is beneficial for training models on small-scale datasets.

## How To Use

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("Langboat/mengzi-bert-base")
model = BertModel.from_pretrained("Langboat/mengzi-bert-base")
```

## Conclusion

This technical report presents our exploration of training lightweight language model called Mengzi, which shows remarkable performance improvements compared with the same-sized or even largerscale models. A series of pre-training and fine-tuning strategies have been verified to be effective for improving model benchmark results. Experimental results show that Mengzi achieves state-of-theart performance with carefully designed training strategies. Without the modification of the model architecture, Mengzi is easy to be deployed as a powerful alternative to existing PLMs.


