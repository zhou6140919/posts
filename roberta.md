---
title: RoBERTa
author: Zhou Tong
date: 2021-11-02 21:00:11
tags: [nlp, papers]
categories: nlp
mathjax: true
---


# A Robustly Optimized BERT Pretraining Approach

<!-- more -->

See [original paper](https://arxiv.org/abs/1907.11692) for more details.

## Problem

Self-training methods such as ELMo, GPT, BERT, XLM, XLNet have brought significant performance gains, but it can be challenging to determine which aspects of the methods contribute the most. Training is computationally expensive, limiting the amount of tuning that can be done, and is often done with private training data of varying sizes, limiting our ability to measure the effects of the modeling advances.

## Solution

A replication study of BERT pre-training, which includes a careful evaluation of the effects of hyperparameter tuning and training set size.

The authors find that BERT was **significantly undertrained** and propose an improved recipe for training BERT models, which the authors call RoBERTa, that can match or exceed the performance of all of the post-BERT methods.

Modifications:
1. training the model longer, with bigger batches, over more data
2. removing the NSP task;
3. training on longer sequences;
4. dynamically changing the masking pattern applied to the training data;
5. collected a new large dataset CC-NEWS.

## Background

### Setup

BERT: takes a concatenation of two segments. $[CLS], x_1, . . . , x_N,[SEP], y_1, . . . , y_M,[EOS]$

where M + N < T

tasks: MLM & NSP

### Optimization

Adam using the following parameters: $β_1 = 0.9, β_2 = 0.999, \epsilon = 1e-6\ and\ L2\ weight\ decay\ of\ 0.01$

warmup: 10,000 steps to a peak value of 1e-4 then linearly decayed.

pre-trained for 1M steps with mini-batches 256 sequences of maximum length 512 tokens. 16GB data.

## Experiment

1. Setting $β_2 = 0.98$ helps to improve stability when training with large batch sizes.
2. Mix precision floating point arithmetic training.
3. Larger datasets 160GB
  - BOOKCORPUS plus English WIKIPEDIA
  - CC-NEWS
  - OPENWEBTEXT
  - STORIES

### Evaluation

**GLUE**

**SQuAD**

**RACE**

## Training Procedure Analysis

### Static vs. Dynamic Masking

**Static Masking** The original BERT implementation performed masking once during data preprocessing, resulting in a single static mask. To avoid using the same mask for each training instance in every epoch, training data was duplicated 10 times so that each sequence is masked in 10 different ways over the 40 epochs of training. Thus, each training sequence was seen with the same mask four times during training.

**Dynamic Masking** The authors generate the masking pattern every time the authors feed a sequence to the model. This becomes crucial when pretraining for more steps or with larger datasets.
![dynamic-masking](/images/dynamic-masking.jpg)

**Results** slightly better than static masking.

### Input Format

![input performance](/images/input-performance.jpg)

**Results** 

1. Using individual sentences hurts performance on downstream tasks, which the authors hypothesize is because the model is not able to learn long-range dependencies.
2. Training with blocks of text from a single document, removing the NSP loss matches or slightly improves downstream task performance.
3. Restricting sequences to come from a single document performs slightly better than packing sequences from multiple documents.

However, because the DOC-SENTENCES format results in variable batch sizes, the authors use FULL-SENTENCES in the remainder of the following experiments.

FULL-SENTENCES: Each input is packed with full sentences sampled contiguously from one or more documents, such that the total length is at most 512 tokens. Inputs may cross document boundaries. When the authors reach the end of one document, the authors begin sampling sentences from the next document and add an extra separator token between documents. The authors remove the NSP loss.

### Training with large batches

Past work in Neural Machine Translation has shown that training with very large mini-batches can both improve optimization speed and end-task performance when the learning rate is increased appropriately.
[Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187)

Recent work has shown that BERT is also amenable to large batch training.
LAMB: [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)

![large batches](/images/large-batches.jpg)

### Text Encoding

BPE: [Byte-Pair Encoding](https://arxiv.org/abs/1508.07909)

[GPT-2](https://arxiv.org/abs/1609.08144) introduced a clever implementation of BPE that uses bytes instead of unicode characters as the base subword units. Using bytes makes it possible to learn a subword vocabulary of a modest size (50K units) that can still encode any input text without introducing any "unknown" tokens.

Training BERT with a larger byte-level BPE vocabulary containing 50K subword units, without any additional preprocessing or tokenization of the input.


## RoBERTa

**R**obustly **o**ptimized **BERT** **a**pproach

Configuration:
- dynamic masking
- FULL-SENTENCES without NSP loss
- large mini-batches
- a larger byte-level BPE

![pretrain comparison](/images/pretrain-comparison.jpg)

The authors begin by training RoBERTa following the $BERT_{LARGE}$ architecture (L = 24, H = 1024, A = 16, 355M parameters).

Even the longest-trained model does not appear to overfit the data and would likely benefit from additional training.

Crucially, RoBERTa uses the same masked language modeling pretraining objective and architecture as $BERT_{LARGE}$, yet consistently outperforms both $BERT_{LARGE}$ and $XLNet_{LARGE}$. This raises questions about the relative importance of model architecture and pretraining objective, compared to more mundane details like dataset size and training time that the authors explore in this work.


## Conclusion

Our improved pretraining procedure, which the authors call RoBERTa, achieves state-of-the-art results on GLUE, RACE and SQuAD, without multi-task finetuning for GLUE or additional data for SQuAD. These results illustrate the importance of these previously overlooked design decisions and suggest that BERT’s pretraining objective remains competitive with recently proposed alternatives.

## References

 - [RoBERTa: A Robustly Optimized BERT Pretraining Approach - Liu et al., 2019](https://arxiv.org/abs/1907.11692)