---
title: OneIE
author: Zhou Tong
date: 2021-11-16 21:14:11
tags: [nlp, papers]
categories: nlp
mathjax: true
---
# A Joint Neural Model for Information Extraction with Global Features

<!-- more -->

See [Original Paper](https://aclanthology.org/2020.acl-main.713) for more details.

## Introduction

**Problem** Most existing joint neural models for Information Extraction (IE) use local task-specific classifiers to predict labels for individual instances (e.g., trigger, relation) regardless of their interactions. Early efforts typically perform IE in a pipelined fashion, which leads to the error propagation problem and disallows interactions among components in the pipeline.

![error made by local classifier without global constraints](/images/error_made_by_local_classifier.jpg)

The model should be able to avoid such mistakes if it is capable of learning and leveraging the fact that it is unusual for an ELECT event to have two PERSON arguments.

**Solution** ONEIE aims to extract the globally optimal IE result as a graph from an input sentence. ONEIE performs end-to-end IE in four stages:
1. Encoding a given sentence as contextualized word representations;
2. Identifying entity mentions and event triggers as nodes;
3. Computing label scores for all nodes and their pairwise links using local classifiers;
4. Searching for the globally optimal graph with a beam decoder. The authors incorporate(纳入) global features to capture the cross-subtask and cross-instance interactions.

IE is a complex task comprised of(由……构成) a wide range of subtasks:
- named, nominal(名词), pronominal(代词) mention extraction
- entity linking
- entity coreference(共指) resolution
- relation extraction
- event extraction
- event coreference resolution

Instead of predicting separate knowledge elements using local classifiers, ONEIE aims to extract a globally optimal information network for the input sentence.

**Results** Experiments show that our framework achieves comparable or better results compared to the state-of-the-art end-to-end architecture [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://arxiv.org/abs/1909.03546).

ONEIE is the first end-to-end neural IE framework that explicitly models cross-subtask and cross-instance interdependencies and predicts the result as a unified graph instead of isolated knowledge elements.

ONEIE does not rely on language-specific features, it can be rapidly applied to new languages.

## Task

Input: sentence
Output: information network representation, where entity mentions and event triggers are represented as nodes and relations are represented as edges.

![ie illustration](/images/ie_illustration.jpg)

### Entity Extraction

This task aims to identify entity mentions in text and classify them into pre-defined entity types.

A mention can be a name, nominal or pronoun. For example, “Kashmir region” should be recognized as a location (LOC) named entity mention

### Relation Extraction

This is the task of assigning a relation type to an ordered pair of entity mentions. For example, there is a PART-WHOLE relation between “Kashmir region” and “India”.

### Event Extraction

This task entails identifying event triggers (the words or phrases that most clearly express event occurrences) and their arguments (the words or phrases for participants in those events) in unstructured texts and classifying these phrases, respectively, for their types and roles. For example, the word “injured” triggers an INJURE event and “300” is the VICTIM argument.

Given an input sentence, our goal is to predict a graph G = (V, E), where V and E are the node and edge sets respectively. Each node $v_i = 〈a_i, b_i, l_i〉 ∈ V$ represents an entity mention or event trigger, where a and b are the start and end word indices, and l is the node type label. Each edge $e_ij = 〈i, j, l_ij〉 ∈ E$ is represented similarly, whereas i and j denote the indices of involved nodes. For example, in Figure 2, the trigger “injured” is represented as 〈7, 7, INJURE〉, the entity mention “Kashmir region” is represented as 〈10, 11, LOC〉, and the event-argument edge between them is 〈2, 3, PLACE〉.

## Approach

There are four steps to extract the information network from a given sentence: encoding, indentification, classification, decoding.

The authors encode the input sentence using a pre-trained BERT encoder and identify entity mentions and event triggers in the sentence. After that, the authors compute the type label scores for all nodes and pairwise edges among them. During decoding, we explore possible information networks for the input sentence using beam search and return the one with the highest global score.

### Encoding

Given an input sentence of L words, the authors obtain the contextualized representation $x_i$ for each word using a pre-trained BERT encoder. If a word is split into multiple word pieces (e.g., Mondrian → Mon, ##dr, ##ian), we use the average of all piece vectors as its word representation. While previous methods typically use the output of the last layer of BERT, our preliminary(初步的) study shows that enriching word representations using the output of the third last layer of BERT can substantially improve the performance on most subtasks.

### Identification

1. The authors use a feedforward network FFN to compute a score vector $\hat{y}_i = FFN(x_i)$ for each word, where each value in $\hat{y}_i$ represents the score for a tag in a target tag set. The authors use the BIO tag scheme, in which the prefix B- marks the beginning of a mention, and I- means inside of a mention. A token not belonging to any mention is tagged with O.
2. the authors use a conditional random fields (CRFs) layer to capture the dependencies between predicted tags (e.g., an I-PER tag should not follow a B-GPE tag)
3. Similar to [Named Entity Recognition with Bidirectional LSTM-CNNs](https://arxiv.org/abs/1511.08308), the authors calculate the score of a tag path $\hat{z} = {\hat{z}\_{1},...,\hat{z}\_{L}}$ as $s(X,\hat{z}) = \sum_{i=1}^{L} \hat{y}\_{i,\hat{z}\_{i}} + \sum_{i=1}^{L+1} A\_{\hat{z}\_{i-1},\hat{z}\_{i}}$ where $X = {x\_{1},...,x\_{L}}$ is the contextualized representations of the input sequence, $\hat{y}\_{i,\hat{z}\_{i}}$ is the $\hat{z}\_{i}$ -th component of the score vector $\hat{y}\_{i}$, and $A\_{\hat{z}\_{i-1},\hat{z}\_{i}}$ is the $ (\hat{z}\_{i-1},\hat{z}\_{i}) $ entry in matrix $A$ that indicates the transition score from tag $\hat{z}\_{i-1}$ to $\hat{z}\_{i}$.
    The weights in A are learned during training.
4. The authors append two special tags start and end to the tag path as $\hat{z}_0$ and $\hat{z}_L+1$ to denote the start and end of the sequence.
5. At the training stage, the authors maximize the log-likelihood of the gold-standard tag path as 

    $$
    log\ p(z|X) = s(X,z) - log\sum_{\hat{z}\in Z}^{} e^{s(X,\hat{z})}
    $$

    where $Z$ is the set of all possible tag paths for a given sentence. Thus, the authors define the identification loss as $\mathcal{L} ^I = - log\ p(z\|X)$.

Note that the authors do not use types predicted by the taggers. Instead, we make a joint decision for all knowledge elements at the decoding stage to prevent error propagation and utilize their interactions to improve the prediction of node type.

### Classification

**???**

To obtain the label score vector for the edge between the i-th and j-th nodes, the authors concatenate their span representations and calculate the vector.

For each task, the training objective is to minimize the cross-entropy loss.

### Global Features

- Cross-subtask interactions.
    “A civilian aid worker from San Francisco was killed in an attack in Afghanistan.” A local classifier may predict “San Francisco” as a VICTIM argument because an entity mention preceding “was killed” is usually the victim despite the fact that a GPE is unlikely to be a VICTIM. To impose such constraints, the authors design a global feature as shown in the figure to evaluate whether the structure DIE-VICTIM-GPE exists in a candidate graph.
    ![cross-subtask interactions](/images/cross-subtask-interactions.jpg)
- Cross-instance interactions.
    “South Carolina boy, 9, dies during hunting trip after his father accidentally shot him on Thanksgiving Day.” It can be challenging for a local classifier to predict “boy” as the VICTIM of the ATTACK event triggered by “shot” due to the long distance between these two words. However, as shown in the figure, if an entity is the VICTIM of a DIE event, it is also likely to be the VICTIM of an ATTACK event in the same sentence.
    ![cross-instance interactions](/images/cross-instance-interactions.jpg)

![global feature categories](/images/global-feature-categories.jpg)

1. Given a graph G, the authors represent its global feature vector as $f_G = {f_1(G), ..., f_M (G)}$, where M is the number of global features and fi(·) is a function that evaluates a certain feature and returns a scalar. For example, 
    ![f1](/images/f1.jpeg)
    <!-- $$
    fi(G) = \left\{  
                \begin{array}{**lr**}  
                1, & G\ has\ multiple\ ATTACK\ events, \\  
                0, & otherwise.   
                \end{array}  
    \right.
    $$ -->

2. ONEIE learns a weight vector $u ∈ \mathbb{R}_M$ and calculates the global feature score of G as the dot product of $f_G$ and $u$. The authors define the global score of G as the sum of its local score and global feature score, namely

    $$
    s(G) = s'(G) + uf_G
    $$

3. The authors make the assumption that the gold-standard graph for a sentence should achieve the highest global score. Therefore, the authors minimize the following loss function $\mathcal{L} ^G = s(\hat{G})- s(G)$ where $\hat{G}$ is the graph predicted by local classifiers and $G$ is the gold-standard graph.

4. The authors optimize the follwing joint objective function during training. $\mathcal{L}= \mathcal{L} ^I + \sum_{t\in T}^{} \mathcal{L} ^t + \mathcal{L} ^G$

### Decoding

**???**

The basic idea is to calculate the global score for each candidate graph and select the one with the highest score. However, exhaustive search is infeasible(不可行的) in many cases as the size of search space grows exponentially(呈几何级数地) with the number of nodes. Therefore, the authors design a beam search-based decoder.

After the last step, the authors return the graph with the highest global score as the information network for the input sentence.

## Experiments

### Data

ACE2005 dataset:
- ACE05-R includes named entity and relation annotations
- ACE05-E includes entity, relation, and event annotations

The authors keep 7 entity types, 6 coarse-grained(粗粒度) relation types, 33 event types, and 22 argument roles.

### Setup

The authors optimize our model with BertAdam for 80 epochs with a lr of 5e-5 and weight decay of 1e-5 for BERT, and a lr of 1e-3 and weight decay of 1e-3 for other parameters.

### Remaining Challenges

- Need background knowledge
- Rare words
- Multiple types per trigger
- Need syntactic structure
- Uncertain events and metaphors

## References

- [A Joint Neural Model for Information Extraction with Global Features, Lin et al., 2020](https://aclanthology.org/2020.acl-main.713.pdf)