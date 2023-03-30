#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

# #### Load an example image

raw_image = Image.open("../docs/_static/merlion.png").convert("RGB")


# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# #### Load BLIP large captioning model finetuned on COCO


# we associate a model with its preprocessors to make it easier for inference.
print("开始加载模型..., 设备是: ", device)
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
)
# uncomment to use base model
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="base_coco", is_eval=True, device=device
# )
vis_processors.keys()


# #### prepare the image as model input using the associated processors


image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


# #### generate caption using beam search


res = model.generate({"image": image})
print(res)

# #### generate multiple captions using nucleus sampling


# due to the non-determinstic nature of necleus sampling, you may get different captions.
# 由于核采样(nucleus sampling)的非确定性性质,你可能会得到不同的图像描述(captions) 使用核采样(use_nucleus_sampling=True)和获取3个描述(num_captions=3),对输入图像(image)进行描述生成
# 打印生成的3个描述(print(res))
res = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)
print(res)
