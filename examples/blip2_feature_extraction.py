#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image

from lavis.models import load_model_and_preprocess


# 加载一张图片，和一段文字
raw_image = Image.open("../docs/_static/merlion.png").convert("RGB")
caption = "a large fountain spewing water into the air"

# 设置要使用的设备
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# 加载模型
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
text_input = txt_processors["eval"](caption)
sample = {"image": image, "text_input": [text_input]}


# #### Multimodal features

features_multimodal = model.extract_features(sample)
print(features_multimodal.multimodal_embeds.shape)
# torch.Size([1, 32, 768]), 32 is the number of queries


# #### Unimodal features


features_image = model.extract_features(sample, mode="image")
features_text = model.extract_features(sample, mode="text")
print(features_image.image_embeds.shape)
# torch.Size([1, 32, 768])
print(features_text.text_embeds.shape)
# torch.Size([1, 12, 768])

# #### Normalized low-dimensional unimodal features

# low-dimensional projected features
print(features_image.image_embeds_proj.shape)
# torch.Size([1, 32, 256])
print(features_text.text_embeds_proj.shape)
# torch.Size([1, 12, 256])
similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
print(similarity)
# tensor([[0.3642]])

