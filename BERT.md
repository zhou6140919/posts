---
title: BERT
author: Zhou Tong
date: 2021-10-27 23:37:54
tags: [nlp, papers]
categories: nlp
cover: https://www.codemotion.com/magazine/wp-content/uploads/2020/05/bert-google.png
mathjax: true
---

BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

<!-- more -->

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

See [original paper](https://arxiv.org/abs/1810.04805) for more details.

## Abstract

BERT = Bidirectional Encoder Representations from Transformers

BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

BERT can be finetuned with just one additional ouput layer for a wide range of tasks.

## Introduction

Two strategies for applying pre-trained language representations to downstream tasks.
- feature-based: ELMo ---- contextual representation: concatenation of left-to-right and right-to-left representations.
- fine-tuning: GPT ---- can only attend to previous tokens in the self-attention layers.

Same objective function during pre-training. They use unidirectional language models to learn general language representations.

**Problem: Current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches. Because the standard language models are unidirectional.**

**Solution: BERT alleviates the unidirectionality constraint by using a "masked language model" (MLM) pre-training objective. The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context.**
Also jointly pre-trains text-pair representations by a "next sentence prediction" task.

The code and pre-trained models are available at [https://github.com/google-research/bert](https://github.com/google-research/bert)

## BERT

### Model Architecture

A multi-layer bidirectional Transformer encoder.

L: the number of layers

H: the hidden size

A: the number of self-attention heads

**BERT-base**: L=12, H=768, A=12

**BERT-large**: L=24, H=1024, A=16

The feed-forward/filter size is 4H, 3072 for H=768 and 4096 for H=1024

Throughout this work, a "sentence" can be an arbitrary(任意的) span of contiguous text, rather than an actual linguistic sentence.

A "sequence" refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

### Embedding

[WordPiece](https://arxiv.org/abs/1609.08144) embeddings with a 30,522 token vocabulary.
The first token of every sequence is always a special classification token [CLS].

Separate sentences with a special token [SEP], and add a learned embedding to every token indicating whether it belongs to sentence A or sentence B.

![bert-pretraining-embedding](/images/bert-pretraining-embedding.jpg)

***The input embedding is the sum of token embedding, segment embedding and position embedding.***

### Pre-training

Using two unsupervised tasks.

#### Task 1: MLM

Mask some percentage of the input tokens at random, and then only predict those mased tokens rather than reconstructing the entire input. (== cloze task).

To mitigate(减轻) the mismatch between pre-training and fine-tuning, the authors replace the chosen token with 80% [mask], 10% a random token, 10% unchanged token.
Using crossentropy loss to predict the original token.

#### Task 2: NSP

Many downstream tasks are based on relationships between two sentences, which is not directly captured by language modeling.

In order to train a model that understands sentence relationships, the authors pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. 50% -- 50% IsNext

There is a better SOP task right now.

#### Data

- BooksCorpus 800M words
- Wikipedia 2,500M words

### Fine-tuning

end-to-end training.

**Input:**
+ single sentence
+ text pairs
  - sentence pairs in paraphrasing
  - hypothesis-premise pairs in entailment
  - question-passage pairs in question answering
  - a degenerate text-∅ pair in text classification or sequence tagging

**Output:**
  - token representations are fed into an ouput layer for token-level tasks, such as sequence tagging or question answering.
  - [CLS] representation is fed into an output layer for classification, such as entailment or sentiment analysis.

## Experiments

### GLUE: The General Language Understanding Evaluation benchmark

The only new parameters introduced during fine-tuning are classification layer weights $W \in \mathbb{R} ^{K \times H}$, where $K$ is the number of labels.
The authors use the final hidden vector $C \in \mathbb{R} ^H$ corresponding to the first input token ([CLS]) as the aggregate representation.
The authors compute a standard classification loss with C and W.

$$log(softmax(CW^T))$$

ps: Log Softmax is advantageous over softmax for numerical stability, optimisation and heavy penalisation for highly incorrect class.
  - Penalises Larger error: More heavy peanlty for being more wrong.
  - Cheaper Model Training Cost: The authors can extrapolate it over all weight W and the authors can easily see that the log-softmax is simpler and faster.

### SQuAD v1.1

The Stanford Wuestion Answering Dataset is a collection of 100k crowd-sourced question/answer pairs.

Given a question and a passage from Wikipedia containing the answer, the task is to predict the answer text span in the passage.

Start vector $S \in \mathbb{R} ^H$ and end vector $E \in \mathbb{R} ^H$.

The probability of word i being the start of the answer span is computed as a dot product between $T_i$ and $S$ followed by a softmax over all of the words in the paragraph:

$$
P_i = \frac{e^{S \cdot T_i}}{\sum e^{S \cdot T_j}} 
$$

For the end of the answer span, the same formula is used.

### SQuAD v2.0

The authors treat questions that do not have ana answer as having an answer span with start and end at the [CLS] token.

### SWAG

The Situations With Adversarial Generations dataset contains 113k sentence-pair completion examples that evaluate grounded common-sense inference.

Given a sentence, the task is to choose the most plausible continuation among four choices.

## Conclusion

Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from deep unidirectional architectures. Our major contribution is further generalizing these findings to deepbidirectionalarchitectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.

## Reference

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Jacob Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
- [The Illustrated BERT, ELMo, and co. - Jay Alammar](https://jalammar.github.io/illustrated-bert/)