---
title: Financial NLP Platform
author: Zhou Tong
date: 2022-11-30 12:30:00
tags: [nlp, projects]
categories: nlp
cover: https://s2.wp.com/wp-content/themes/a8c/jetpackme-new/images/2021/social-growth.jpg
feature: true
---

Langboat zero-shot information extraction platform can extract entities with specific meanings from texts. For example, in the scenario of sorting out punishment announcements, it is necessary to extract the entity information such as law enforcement agencies, punishment objects, and punishment amounts, and quickly structure a large number of texts to improve efficiency.

<!-- more -->

# Financial Zero-shot Information Extraction Platform

:::warning Links

[Product Trial](https://cognitive.langboat.com/product/finance-extraction)

:::

## Introduction

This platform can extract entities with specific meaning in text. 
For example, in the scene of sorting out punishment announcements, 
it is necessary to extract entity information such as law enforcement agencies, punishment objects, and punishment amounts, 
and quickly structure a large amount of text to improve efficiency.

This platform supports users to establish entity types and customize personalized information extraction models according to their own needs.
Different from the traditional AI model creation method that needs to manually annotate hundreds or thousands of sample data, 
on the zero-shot platform, it only takes three simple steps to realize the customization of the information extraction model.

![progress](https://cdn.langboat.com/image/zero-shot/progress.png)

## Concepts

### Schemas

The schema is the entity information that needs to be output in the information extraction task.

For example, if you need to extract "January 1st" and "Company A" from "Company A acquired Company B on January 1st",
then you can create two new schemas: "Time" and "Purchaser".

### Prompts

Model prompts are additional words that help the model locate key information when inputting text to be extracted;
generally speaking, they are "questions" posed to the model.

For example, if you need to extract "time" from "Company A acquired Company B on January 1st",
you can enter the model prompt "What is the time mentioned above?"

For different schemas, different model prompts also have different extraction effects.
You can choose the most appropriate model prompt for each schema to ensure the accuracy of the model.

## Example Code in Python

1. Press the "Product Trail" button and register in the product control panel
2. Replace your API key in the code below

> ```python example.py
> import base64
> import datetime
> import hashlib
> import hmac
> import json
> import random
> import requests
> 
> 
> class LangboatOpenClient:
>     """???????????????????????????"""
> 
>     def __init__(self,
>                  access_key: str,
>                  access_secret: str,
>                  url: str = "https://open.langboat.com"):
>         self.access_key = access_key
>         self.access_secret = access_secret
>         self.url = url
> 
>     def _build_header(self, query: dict, data: dict) -> dict:
>         accept = "application/json"
>         # 1. body MD5 ??????
>         content_md5 = base64.b64encode(
>             hashlib.md5(
>                 json.dumps(data).encode("utf-8")
>             ).digest()
>         ).decode()
>         content_type = "application/json"
>         gmt_format = '%a, %d %b %Y %H:%M:%S GMT'
>         date = datetime.datetime.utcnow().strftime(gmt_format)
>         signature_method = "HMAC-SHA256"
>         signature_nonce = str(random.randint(0, 65535))
>         header_string = f"POST\n{accept}\n{content_md5}\n{content_type}\n" \
>                         f"{date}\n{signature_method}\n{signature_nonce}\n"
> 
>         # 2. ?????? queryToSign
>         queries_str = []
>         for k, v in sorted(query.items(), key=lambda item: item[0]):
>             if isinstance(v, list):
>                 for i in v:
>                     queries_str.append(f"{k}={i}")
>             else:
>                 queries_str.append(f"{k}={v}")
>         queries_string = '&'.join(queries_str)
>         # 3.?????? stringToSign
>         sign_string = header_string + queries_string
>         # 4.?????? HMAC-SHA256 + Base64
>         secret_bytes = self.access_secret.encode("utf-8")
>         # 5.????????????
>         signature = base64.b64encode(
>             hmac.new(secret_bytes, sign_string.encode(
>                 "utf-8"), hashlib.sha256).digest()
>         ).decode()
>         res = {
>             "Content-Type": content_type,
>             "Content-MD5": content_md5,
>             "Date": date,
>             "Accept": accept,
>             "X-Langboat-Signature-Method": signature_method,
>             "X-Langboat-Signature-Nonce": signature_nonce,
>             "Authorization": f"{self.access_key}:{signature}"
>         }
>         return res
> 
>     def inference(self, queries: dict, data: dict) -> (int, dict):
>         """
>         ??????
>         :param queries: query ??????
>         :param data: request body ??????
>         :return: response status, response body to json
>         """
>         headers = self._build_header(queries, data)
>         response = requests.post(
>             url=self.url, headers=headers, params=queries, json=data)
>         return response.status_code, response.json()
> 
> 
> if __name__ == '__main__':
>     _access_key = '<your-access-key>'
>     _access_secret = '<your-access-secret>'
> 
>     client = LangboatOpenClient(
>         access_key=_access_key,
>         access_secret=_access_secret
>     )
> 
>     _queries = {
>         "action": "zeroShotTask",
>         "task_id": "39",
>     }
> 
>     _data = {
>         "sample": "??????????????????(9???10???)??????????????????????????????.???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????"}
> 
>     status_code, result = client.inference(_queries, _data)
>     print("response status:", status_code)
>     print("response json:", json.dumps(result, ensure_ascii=False, indent=2))
> ```

```bash
python example.py
```