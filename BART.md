---
title: BART
author: Zhou Tong
date: 2022-12-10 13:48:02
tags: [nlp, papers]
categories: nlp
feature: true
mathjax: true
---

# BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

<!-- more -->

See [original paper](https://arxiv.org/abs/1910.13461) for more details.

## Introduction

BART is trained by 
    - corrupting text with an arbitrary noising function
    - learning a model to reconstruct the original text

BART is particularly effective when fine-tuned for text generation but also works well for comprehension tasks.

## Model

### Architecture

Standard sequence-to-sequence Transformer architecture

BART = BERT + GPT(ReLU -> GeLU and initialize parameters from $\mathcal{N}(0, 0.02)$)

Model differences:
     - each layer of the decoder additionally performs cross-attention over the final hidden layer of the encoder (as in the transformer model);
     - BERT uses an additional feed-forward network before word prediction which BART does not.

### Pre-training

**Objective** The cross-entropy loss between the decoder's output and the original document.

**Token Masking** Following BERT, random tokens are sampled and replaced with $[MASK]$ token.

**Token Deletion** Random tokens are deleted from the input. The model must decide which positions are missing inputs.

**Text Infilling** A number of text spans are sampled with span lengths drawn from a Poisson distribution ($\lambda = 3$). Each span is replaced with a single $[MASK]$ token. Not like SpanBERT, text infilling teaches the model to predict how many tokens are missing from a span.

**Sentence Permutaion** A document is divided into sentences based on full stops, and these sentences are shuffled in a random order.

**Document Rotation** A token is chosen uniformly at random, and the document is rotated so that it begins with this token. This task trains the model to identify the start of the document.

## Fine-tuning

### Token Classification Tasks

SQuAD: the authors feed the complete document into the encoder and decoder and use the top hidden state of the decoder as a representation for each word. This representation is used to classify the token.

### Sequence Generation Tasks

Information is copied from the input but manipulated, which is closely related to the denoising pre-training objective.

### Machine Translation

The authors replace BART’s encoder embedding layer with a new randomly initialized encoder. The model is trained end-to-end, which trains the new encoder to map foreign words into an input that BART can de-noise to English. The new encoder can use a separate vocabulary from the original BART model.

1. freeze most of BART parameters and only update the randomly initialized source encoder, the BART positional embeddings, and the self-attention input projection matrix of BART’s encoder first layer;
2. train all model parameters for a small number of iterations.

## Comparing Pre-training Objectives

### Comparison Objectives

**Language Model** BART decoder without cross-attention.

**Permuted Language Model** Sample 1/6 of the tokens and generate them in a random order autoregressively. Didn't use the relative positional embeddings or attention across segements from XLNet.

**Masked Language Model** Following BERT, the authors replace $15\\%$ of tokens with $[MASK]$ token.

**Multitask Masked Language Model**  Additional sef-attention masks as in UniLM.

**Masked Seq-to-Seq** Inspired by MASS, the authors mask a span containing $50\\%$ of tokens.

For the Permuted LM, Masked LM and Multitask Masked LM, the authors use two-stream attention to efficiently compute likelihoods of the output part of the sequence.
