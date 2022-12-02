---
title: Hexo Instructions
author: Zhou Tong
date: 2022-12-02 22:24:59
tags: [hexo]
categories: frontend
feature: true
---

## Quick Start

Create a new post
```bash
hexo new "My New Post"
```

Create a draft which will not be published
```bash
hexo new draft "My New Draft"
```

You can simply use `hexo new 'title'` to create a new file.

The default type of this file can be changed in `_config.yml`.
```yml
# Writing
new_post_name: :title.md # File name of new posts
default_layout: draft
```

## Tag Plugins

[Documentation](https://hexo.io/docs/tag-plugins)

### Code Block

{% codeblock lang:javascript%}
    alert("Hello World!");
    var a = "hello";
{% endcodeblock %}

```python python
def hello():
    print("Hello World!")
```
### Blockquote
{% blockquote %}
Hello World !
{% endblockquote %}
### Youtube Video
{% youtube hY7m5jjJ9mM %}

### Pullquote
{% pullquote %}
content
{% endpullquote %}
### Jsfiddle
{% jsfiddle shorttag html, 500, 100 %}

### Include Posts

```
{% post_path filename %}
{% post_link filename [title] [escape] %}
```
{% post_link hello-world %}