#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/3/24 10:31
# @File  : lora_utils.py
# @Author:  johnson
# @Desc  : Lora的配置，用于微调模型
from peft import (
    LoraConfig,         #LoraConfig是Lora的配置
    LoraModel,     # get_peft_model是获取peft模型
    get_peft_model_state_dict,  #get_peft_model_state_dict是获取peft模型的状态字典
)

def get_lora_model(model, config):
    """
    把给定模型变成Lora模型，用于微调
    根据给定的config，生成一个Lora模型
    """
    model_cfg = config.model_cfg
    enable_lora = model_cfg.enable_lora
    load_finetuned = model_cfg.load_finetuned
    if not enable_lora:
        return model
    print("转变模型为Lora模型")
    if not load_finetuned:
        print(f"启用了lora模型，建议先加载已经微调好的模型，即load_finetuned参数设置为True,不要随机初始化模型")
    lora_target_modules = model_cfg.lora_target_modules
    lora_r = model_cfg.lora_r
    lora_alpha = model_cfg.lora_alpha
    lora_dropout = model_cfg.lora_dropout
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    lora_model = LoraModel(lora_config, model)
    model.print_trainable_parameters() # 打印模型的参数
    return lora_model