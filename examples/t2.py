#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/3/24 10:24
# @File  : t2.py
# @Author: 
# @Desc  :

from transformers import pipeline, set_seed

set_seed(32)
generator = pipeline('text-generation', model="facebook/opt-2.7b", do_sample=True)
result = generator("你好，我是")
print(result)