#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/3/24 10:31
# @File  : lora_utils.py
# @Author:  johnson
# @Desc  : Lora的配置，用于微调模型
import os
import json
import torch
from peft import (
    LoraConfig,         #LoraConfig是Lora的配置
    LoraModel,     # get_peft_model是获取peft模型
    get_peft_model_state_dict,  #get_peft_model_state_dict是获取peft模型的状态字典
)
WEIGHTS_NAME = "adapter_model.bin"
CONFIG_NAME = "adapter_config.json"

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
    print("模型结构是：")
    print(lora_model)
    return lora_model

def save_lora_model(save_directory, model, config, state_dict=None):
    """
    保存Lora模型, runner_base.py中的_save_checkpoint已经能保存模型了，这里不需要了
    """
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
    os.makedirs(save_directory, exist_ok=True)

    # 只保存可训练的权重
    output_state_dict = get_peft_model_state_dict(model, state_dict)
    torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME))
    # 保存配置
    output_path = os.path.join(save_directory, CONFIG_NAME)
    # save it
    with open(output_path, "w") as writer:
        writer.write(json.dumps(config, indent=2, sort_keys=True))