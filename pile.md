---
title: PILE
author: Zhou Tong
date: 2022-12-24 00:43:49
tags: [nlp, papers]
categories: nlp
feature: true
mathjax: true
---

# The Pile: An 800GB Dataset of Diverse Text for Language Modeling

<!-- more -->

See [original paper](http://arxiv.org/abs/2101.00027) for more details.

## Introduction

**Motivation** Recent work has demonstrated that increased training dataset diversity improves general cross-domain knowledge and downstream generalization capability for large-scale language models.

**Contribution** 

1. Pile: an 825GiB English text corpus targeted at training large-scale language models. The Pile is constructed from 22 diverse high-quality subsets, many of which derive from academic or professional sources.
2. The introduction of 14 new language modeling datasets, which the authors expect to be of independent interest to researchers.
3. Evaluations demonstrating significant improvements across many domains by GPT-2-sized models trained on this new dataset, compared to training on CC-100 and raw Common Crawl.
4. The investigation and documentation of this dataset, which the authors hope will better inform researchers about how to use it as well as motivate them to undertake similar investigations of their own data.

<button onclick="javascript:window.location.href='https://github.com/EleutherAI/the-pile'">Code Implementation</button>

## Details

<img src="/images/pile-components.png" style="display: block; margin: 0 auto">
<center>Pile Components</center>

Pile-CC is a new filtered subset of Common Crawl with improved extraction quality.





