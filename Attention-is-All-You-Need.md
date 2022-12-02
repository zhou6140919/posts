---
title: Attention is All You Need
author: Zhou Tong
date: 2021-10-21 23:35:24
tags: [nlp, papers]
categories: nlp
feature: true
cover: https://www.hollywoodreporter.com/wp-content/uploads/2018/05/transformers_last_knight_2017_5_copy_-_h_2018.jpg?w=1024
description: The Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. In this work the authors propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
mathjax: true
---

See [Original Paper](https://arxiv.org/abs/1706.03762) for more details.

See [Transformer Codes by Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html) for more details.


# Notes on Attention is All You Need

`Problems:`
Sequence models are based on complex recurrent or convolutional neural networks. RNN aligns the positions to steps in computation time. The inherently(内在地) sequential nature precludes(杜绝) parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.


`Solution:`
The Transformer, based solely on attention mechanisms, dispensing with(摒弃) recurrence and convolutions entirely.
In this work the authors propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.


`Result:`
Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Reached SOTA. 28.4 BLEU on the WMT 2014 en-to-german translation task and 41.8 on the WMT en-to-french translation task.
By parsing both large and limited training data, the Transformer generalizes well to other tasks in English.

---------------------------

LSTM: see [Sequence to Sequence Learning with Neural Networks][1]

Gated recurrent neural networks: see [Empirical evaluation of gated recurrent neural networks on sequence modeling.][2]



**Reached the boundaries of recurrent language models and encoder-decoder architectures.**

RNN: Factor computation along the symbol positions of the input and output sequences. 
`Parallelization precluded.`

Recent works has been achieved significant improvements in computational efficiency through factorization(因数分解) tricks and conditional computation. The fundamental constraint of sequential computation however remains.


Attention mechanism used before transformer are used in conjunction(结合) with a recurrent network.

The number of operations required to relate signals from two arbitrary input or output position grows in the distance between positions. Linearly for ConvS2S and logarithmically for ByteNet. 
`This makes it more difficult to learn dependencies between distant positions.`


In Transformer this is reduced to a constant number of operations, albeit(尽管) at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect the authors counteract(抵消) with Multi-Head Attention.



Self-attention: is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

>是一种将单个序列的不同位置关联起来以计算序列表示的注意力机制。


End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks.

---------------------------

## Model Architecture

Encoder maps an input sequence of symbol representations to a sequence of continuous representations. Decoder then generates an output sequence of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.



![structure of Transformer](http://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png)

This is the structure of Transformer.



The output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model} = 512$.

LayerNorm: see [Layer Normalization][3]


The authors also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i. 

这种掩码与输出嵌入偏移一个位置的事实相结合，确保位置 i 的预测只能依赖于小于 i 位置的已知输出。

### Attention Function

The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility(相关性) function of the query with the corresponding key.

The authors compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V.


$$
Attention(Q,K,V)=softmax((QK^T)/√(d_k))V
$$

Dot product is much faster and more space-efficient, since it can be implemented using highly optimized matrix multiplication code.


The authors suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, the authors scale the dot products by $1/√(d_k )$.

(the variance grows from 1 to dk when q and k dot product.

### Multi-Head Attention

On each of these projected versions of queries, keys and values the authors then perform the attention function in parallel, yielding $d_v$-dimensional output values. These are concatenated and once again projected, resulting in the final values.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

$$
MultiHead(Q,K,V)=Concat(head_1,…,head_h)
$$

$$
where\ head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)
$$

>Cross attention:
>Queries from previous decoder layer, keys and values come from the output of the encoder.
>This allows every position in the decoder to attend over all positions in the input sequence.


>Feed-Forward Networks
>
>$$FFN(x) = max(0, xW_1 + b_1 )W_2 + b_2$$
>
>ReLU activation between two linear transformations. 512->2048->512

>Input embeddings add positional embeddings.
>
>Using sine and cosine functions.

--------------------------

## Advantages

One is the total computational complexity per layer. 


Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.


The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies.


| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| :---: | :---: | :---: | :---: |
| Self-Attention | $O(n^2 · d)$ | $$O(1)$$ | $$O(1)$$ |
| Recurrent | $$O(n · d^2)$$ | $$O(n)$$ | $$O(n)$$ |
| Convolution | $$O(n^2 · d)$$ | $$O(1)$$ | $$O(log_k(n))$$ |
| Self-Attention(Restricted) | $$O(r · n · d)$$ | $$O(1)$$ | $$O(n/r)$$ |


----------------------------------------------

  [1]: https://arxiv.org/abs/1409.3215
  [2]: https://arxiv.org/abs/1412.3555
  [3]: https://arxiv.org/abs/1607.06450


## References
- [Attention is All You Need - Ashish Vaswani, et al.](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer - Alexander M. Rush](https://jalammar.github.io/illustrated-transformer/)
