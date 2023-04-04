#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/4/3 18:02
# @File  : t6.py
# @Author: 
# @Desc  :

# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

input_text = "你好"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
