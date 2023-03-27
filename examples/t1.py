#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/3/22 17:44
# @File  : t1.py
# @Author: 
# @Desc  : mini 数据集测试
import os
import json
def cut_off_data(data_file, data_num, output_file):
    """
    截断数据集,
    """
    # 读取json文件data_file
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 截断数据集
    new_data = data[:data_num]
    # 写入json文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print(new_data[0])
    print(f"截断数据集完成，共{data_num}条数据")

def all_data_file():
    path = "/home/wac/johnson/.cache/lavis/coco/annotations"
    train_file = os.path.join(path, "coco_karpathy_train.json")
    mini_train_file = os.path.join(path, "coco_karpathy_mini_train.json")
    test_file = os.path.join(path, "coco_karpathy_test.json")
    mini_test_file = os.path.join(path, "coco_karpathy_mini_test.json")
    val_file = os.path.join(path, "coco_karpathy_val.json")
    mini_val_file = os.path.join(path, "coco_karpathy_mini_val.json")
    cut_off_data(train_file, 600, mini_train_file)
    cut_off_data(test_file, 100, mini_test_file)
    cut_off_data(val_file, 300, mini_val_file)

if __name__ == '__main__':
    all_data_file()