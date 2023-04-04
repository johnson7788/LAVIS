#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/4/4 10:58
# @File  : t7.py
# @Author: 
# @Desc  :
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load sample image
raw_image = Image.open("docs/_static/merlion.png").convert("RGB")

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
# ask a random question.
# question = "Which city is this photo taken?"
# question = "照片中的城市是哪个？"
question = "照片中有什么内容？"
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
question = txt_processors["eval"](question)
res = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
print(res)