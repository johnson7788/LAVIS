#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/3/22 17:44
# @File  : t1.py
# @Author: 
# @Desc  : mini 数据集测试
from lavis.models import model_zoo
print(model_zoo)
"""
==================================================
Architectures                  Types
==================================================
albef_classification           ve
albef_feature_extractor        base
albef_nlvr                     nlvr
albef_pretrain                 base
albef_retrieval                coco, flickr
albef_vqa                      vqav2
alpro_qa                       msrvtt, msvd
alpro_retrieval                msrvtt, didemo
blip_caption                   base_coco, large_coco
blip_classification            base
blip_feature_extractor         base
blip_image_text_matching       base, large
blip_nlvr                      nlvr
blip_pretrain                  base
blip_retrieval                 coco, flickr
blip_vqa                       vqav2, okvqa, aokvqa
blip2_opt                      pretrain_opt2.7b, pretrain_opt6.7b, caption_coco_opt2.7b, caption_coco_opt6.7b
blip2_t5                       pretrain_flant5xl, pretrain_flant5xl_vitL, pretrain_flant5xxl, caption_coco_flant5xl
blip2_feature_extractor        pretrain, pretrain_vitL, coco
blip2                          pretrain, pretrain_vitL, coco
blip2_image_text_matching      pretrain, pretrain_vitL, coco
pnp_vqa                        base, large, 3b
pnp_unifiedqav2_fid            
img2prompt_vqa                 base
clip_feature_extractor         ViT-B-32, ViT-B-16, ViT-L-14, ViT-L-14-336, RN50
clip                           ViT-B-32, ViT-B-16, ViT-L-14, ViT-L-14-336, RN50
gpt_dialogue                   base

"""