---
title: Mengzi-zero-shot
author: Zhou Tong
date: 2022-08-30 12:30:00
tags: [nlp, projects, sdk]
categories: nlp
cover: http://files.langboat.com/images/mengzicover.png
description: All interface capabilities provided in this project are based on Mengzi-T5-base-MT. This model is a multi task model, which is based on Mengzi-T5-base and trained by using 27 additional datasets and 301 prompts. This project provides the capabilities of entity extraction, semantic similarity, financial relationship extraction, advertisement generation, intention classification in medical field, emotion classification, comment object extraction, news classification, etc.
feature: true
---

# Mengzi-zero-shot

:::warning Links

 - [Model](https://huggingface.co/Langboat/mengzi-t5-base-mt)
 - [SDK Repo](https://github.com/Langboat/mengzi-zero-shot)

:::

All interface capabilities provided in this project are based on Mengzi-T5-base-MT. This model is a multi task model, which is based on Mengzi-T5-base and trained by using 27 additional datasets and 301 prompts. This project provides the capabilities of entity extraction, semantic similarity, financial relationship extraction, advertisement generation, intention classification in medical field, emotion classification, comment object extraction, news classification, etc.

# Navigation
* [Quick Start](#Quick-Start)
* [Interface Capabilities](#Interface-Capabilities)
* [Interface Description](#Interface-Description)

# Quick Start
## Create a new environment
```
conda create -n mengzi_env python=3.7 -y
conda activate mengzi_env
```

## pip install
```bash
pip install mengzi-zero-shot
```

## test examples
```
python test.py
```

# Interface Capabilities

| Interface Capabilities                 | Brief Description                                             |
| ---------------- | ------------------------------------------------------------ |
| [Entity Extraction](#Entity-Extraction) | Extract the named entities in the text and provide the categories of the corresponding entities, such as addresses, book titles, companies, games, governments, movies, names, organizations, positions, attractions, etc. |
| [Semantic Similarity](#Semantic-Similarity)       | Measures the semantic similarity between two texts. |
| [Financial Relationship Extraction](#Financial-Relationship-Extraction)   | Determine which relationship two entities in the text belong to, such as issuing, being issued, capital injection, increase in holdings, etc.            |
| [Advertisement Generation](#Advertisement-Generation)      | Given the keyword information of the product, generate a reasonable advertisement.                    |
| [Medical Field Intent Classification](#Medical-Field-Intent-Classification)  | According to the input text in the medical field, identify user query intentions, such as index interpretation, disease diagnosis, medical advice, treatment plan, etc. |
| [Sentiment Classification](#Sentiment-Classification)          | Classify text into sentiment polarity categories (positive, negative). |
| [Comment Object Extraction](#Comment-Object-Extraction)          | For a given comment text, automatically extract the evaluation objects contained in it.                |
| [News Classification](#News-Classification)          | Classify the input news text, such as agriculture, culture, e-sports, sports, finance, entertainment, tourism, education, finance, military, real estate, automobile, stock, international, etc. |
| [Name Extraction](#Name-Extraction) | Extract the name of the person in the text. |
| [Company Name Extraction](#Company-Name-Extraction) | Extract the company name in the text. |


# Interface Description

## Entity Extraction

Extract the named entities in the text and provide the categories of the corresponding entities, such as addresses, book titles, companies, games, governments, movies, names, organizations, positions, attractions, etc.

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='entity_extraction', 
                   input_string='导致泗水的砭石受到追捧，价格突然上涨。而泗水县文化市场综合执法局颜鲲表示，根据监控')
print(res)
```

Output:

```
"泗水：地址，泗水县文化市场综合执法局：政府，颜鲲：姓名"
```

## Semantic Similarity

Measures the semantic similarity between two texts.

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='text_similarity', 
                   input_string='你好，我还款银行怎么更换',
                   input_string2='怎么更换绑定还款的卡')
print(res)
```

Output:
```
"是"
```

## Financial Relationship Extraction

Enter a piece of text and two entities contained in the text, and determine which relationship the two entities in the text belong to.

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='financial_relationship_extraction', 
                input_string='为打消市场顾虑,工行两位洋股东——美国运通和安联集团昨晚做出承诺,近期不会减持工行H股。',
                entity1='工行',
                entity2='美国运通')
print(res)
```

Output:
```
 "被持股"
```

## Advertisement Generation

Input a description text of a product to automatically generate an effective advertisement.

```python
 from mengzi_zs import MengziZeroShot
 mz = MengziZeroShot()
 mz.load()
 res = mz.inference(task_type='ad_generation', 
                    input_string='类型-裤，版型-宽松，风格-潮，风格-复古，风格-文艺，图案-复古，裤型-直筒裤，裤腰型-高腰，裤口-毛边')
print(res)
```

Output:
```
 "小宽松版型与随性的风格颇具人气，高腰的设计，毛边裤脚，增添潮流气息。考究的做旧质感，洋溢着复古的气息，一款风格随性却不失复古文艺的直筒牛仔裤。,宽松的直筒版型，对身材的包容度较大，穿着舒适无束缚感。高腰的设计，提升腰线，拉长身材比例，打造大长腿的既视感"
```

## Medical Field Intent Classification

Input a piece of medical query text, and identify the user's query intent in the text.

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='medical_domain_intent_classifier', 
                   input_string='呼气试验阳性什么意思')
print(res)
```
Output:

```
 "指标解读"
```

## Sentiment Classification

Input a piece of text, and classify the text containing subjective opinion information into emotional polarity categories (positive, negative).

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='sentiment_classifier', 
                   input_string='房间很一般，小，且让人感觉脏，隔音效果差，能听到走廊的人讲话，走廊光线昏暗，旁边没有什么可吃')
print(res)
```
Output:
```
 "消极"
```

## Comment Object Extraction

Input a piece of text and automatically extract the evaluation objects contained in it.

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='comment_object_extraction', 
                   input_string='灵水的水质清澈，建议带个浮潜装备，可以看清湖里的小鱼。')
print(res)
```
Output:

```
 "灵水"
```

## News Classification

Input a piece of news and identify its category, such as agriculture, culture, e-sports, sports, finance, entertainment, tourism, education, finance, military, real estate, automobiles, stocks, international, etc.

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='news_classifier', 
                   input_string='懒人适合种的果树：长得多、好打理，果子多得都得送邻居吃')
print(res)
```
Output:
```
 "农业"
```

## Name Extraction

Input a piece of text and identify the names of people that appear in it.

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='name_extraction',
		   input_string='我是张三，我爱北京天安门')
print(res)
```
Output:
```
"张三"
```

## Company Name Extraction

Input a text identifying the company name as it appears.

```python
from mengzi_zs import MengziZeroShot
mz = MengziZeroShot()
mz.load()
res = mz.inference(task_type='company_extraction',
		   input_string='就天涯网推出彩票服务频道是否是业内人士所谓的打政策“擦边球”，记者近日对此事求证彩票监管部门。')
print(res)
```
Output:
```
"天涯网"
```