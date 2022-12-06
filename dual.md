---
title: News Monitoring System Based on Event Extraction and Data-to-Text Generation
author: Zhou Tong
date: 2022-06-07 19:55:34
categories: nlp
tags: [nlp, projects]
feature: true
---

A method to improve the effect of event extraction task.

<!-- more -->


# News Monitoring System Based on Event Extraction and Data-to-Text Generation

**Proposed a dual learning algorithm that treats event extraction and text reconstruction as dual tasks and trained both models.**

![dual-algorithm](/images/IMG_1374.PNG)

---

**Used T5 as a generation model to generate pseudo data from structured event information to train the event extraction model, and applied the [GTEE-base](https://arxiv.org/abs/2205.06166) model as the baseline event extraction model.**

![t5-generation](/images/IMG_1373.PNG)

---

**Added reinforcement learning-based and BLEU score-based policy gradient method to the event extraction model's objective to opitimize the model results and achieved SOTA an the ACE05-E, ACE05-E+ and ERE-EN datasets.**

---

**Created an interface of the system.**

![interface](/images/IMG_1375.PNG)
