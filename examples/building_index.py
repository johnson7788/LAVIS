#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/12/7 16:31
# @File  : building_index.py
# @Author:
# @Desc  :  对于LAVIS框架，给所有商品创建encoding向量

import os
import time
import json
import argparse
import hashlib
import logging
import copy
import typing
import base64
import re
import collections
from tqdm import tqdm
import pymongo
import pymysql
import pandas as pd
import numpy as np
import requests
import torch
from omegaconf import OmegaConf
from PIL import Image
from lavis.common.config import Config
import lavis.tasks as tasks
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.comestic_datasets import (ComesticDataset, ComesticEvalDataset)
from lavis.runners.runner_base import RunnerBase

class ComesticCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ComesticDataset
    eval_dataset_cls = ComesticEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cosmetic/defaults_cos.yaml",
        "mini": "configs/datasets/cosmetic/mini_cos.yaml",
    }
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.cache_path = "/home/wac/johnson/.cache/lavis"
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        logging.info(f"数据集")
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)
        is_train = False
        datasets = dict()

        # processors
        vis_processor = (
            self.vis_processors["train"]
            if is_train
            else self.vis_processors["eval"]
        )
        text_processor = (
            self.text_processors["train"]
            if is_train
            else self.text_processors["eval"]
        )
        # create datasets
        test_path = os.path.join(self.cache_path, ann_info["test"]["storage"])
        ann_paths = [test_path]
        vis_path = vis_info.storage
        datasets["test"] = ComesticEvalDataset(
            vis_processor=vis_processor,
            text_processor=text_processor,
            ann_paths=ann_paths,
            vis_root=vis_path,
        )
        return datasets

class MyRunnerBase(RunnerBase):
    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)

    @torch.no_grad()
    def do_predict(self):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on. eg:test
            cur_epoch (int): current epoch. eg: best
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                测试期间，设为False，加载我们提供的权重，而不是训练时的best checkpoint.
        """
        data_loader = self.dataloaders.get("test", None)
        assert data_loader, "data_loader for split test is None."
        results = self.task.evaluation(self.model, data_loader)
        return results

class Inference:
    def __init__(self, time_key, checkpoint_path: str = None, url_id_predict=True):
        """
        :param time_key:
        :type time_key:
        :param checkpoint_path:
        :type checkpoint_path:
        :param url_id_predict:  如果提供url，那么是否使用url中id事先进行过滤
        :type url_id_predict:
        """
        self.time_key = time_key
        self.url_id_predict = url_id_predict
        self.batch_size = 16
        logging.info(f"初始化中")
        #已经存在的图片
        self.path = "/home/wac/johnson/.cache/torch/mmf/data/datasets/retrieval_tmall"
        self.images_dir = os.path.join(self.path,"images")
        #如果给的链接的url的图片不存在，那么自动下载的目录
        self.cache_dir = os.path.join(self.path,"cache")
        self.annotation_path = os.path.join(self.path,"annotations_deploy")
        # 获取urlid
        # 把checkpoint_path变成绝对路径
        checkpoint_path = os.path.abspath(checkpoint_path)
        self.checkpoint = checkpoint_path
        assert self.checkpoint is not None
        assert os.path.exists(self.checkpoint),f"模型文件不存在，请检查"
        self.read_cache_ocr()
        self.batch_accuracy = []
        # node_data 节点数据
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cuda = False
        self._build_model()
        #  读取所有标签
        self.read_label2id(time_key)
    def _build_model(self):
        # 解析配置
        cfg = Config(args)
        logging.info(f"初始化模型和数据集")
        # 设置随机种子
        # 设置日志
        # set after init_distributed_mode() to only log on master.
        # 打印配置
        # 设置任务
        task = tasks.setup_task(cfg)
        datasets_config = cfg.datasets_cfg
        datasets = {}
        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = ComesticCapBuilder(dataset_config)
            dataset = builder.build()
            datasets[name] = dataset
        self.datasets = datasets
        # 设置模型
        #更改finetuned模型路径
        cfg.model_cfg.finetuned = self.checkpoint
        model = task.build_model(cfg)
        model.eval()
        self.model = model
        self.cfg = cfg
        self.task = task
        self.model.to(self.device)
    def encoder(self, data):
        """
        :param data:  list， 每个元素
        :type data:
        :return:
        :rtype:
        """
        #拷贝一份数据集，然修改包含data的数据集
        datasets = copy.deepcopy(self.datasets)
        datasets['comestic_caption']['test'].annotation = data
        runner = MyRunnerBase(
            cfg=self.cfg, job_id="predict_task", task=self.task, model=self.model, datasets=datasets
        )
        results = runner.do_predict()
        return results
    def download(self,image_url, image_path_name, force_download=False, return_exists=False):
        """
        根据提供的图片的image_url，下载图片保存到image_path_name
        :param: force_download: 是否强制下载
        :return_exists: 是否图片已经存在，如果已经存在，那么返回2个True
        """
        # 如果存在图片，并且不强制下载，那么直接返回
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.132 Safari/537.36"
        }
        if os.path.exists(image_path_name) and not force_download:
            if return_exists:
                return True, True
            else:
                return True
        try:
            response = requests.get(image_url, stream=True, headers=headers)
            with open(image_path_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            if return_exists:
                return True, False
            else:
                return True
        except Exception as e:
            print(f"{image_url}下载失败")
            print(f"{e}")
            if return_exists:
                return False, False
            else:
                return False
    def read_cache_ocr(self):
        """
        读取缓存的ocr结果
        :return:
        :rtype:
        """
        ocr_data = mongo_read(database='label', collection='ocr')
        ocr_dict = {}
        for one in ocr_data:
            md5 = one["md5"]
            ocr = one["ocr"]
            ocr_text = ";".join(ocr)
            ocr_dict[md5] = ocr_text
        self.ocr_dict = ocr_dict

    def cal_md5(self,content):
        """
        计算content字符串的md5
        :param content:
        :return:
        """
        # 使用encode
        result = hashlib.md5(content.encode())
        # 打印hash
        md5 = result.hexdigest()
        return md5
    def read_label2id(self, time_key):
        """
        pid到id的映射
        :param time_key: mongo的标签的key
        :return:
        :rtype:
        """
        data = mongo_find(query={"time_key": time_key}, database="label",collection="train_label")
        label_data = {}
        for i in data:
            i.pop("_id")
            label_data.update(i)
        self.product_name2alias = label_data["product_name2alias"]
        self.product_id2names = label_data["product_id2names"]
        self.product_name2brand = label_data["product_name2brand"]
        self.product_name2picture = label_data["product_name2picture"]
        self.product_name2category = label_data["product_name2category"]
        self.product_name2bigcatg = label_data["product_name2bigcatg"]
        # labels_nums = self.check_config["heads"]["product_head"][0]["num_labels"]
        # brand_nums = self.check_config["heads"]["brand_head"][0]["num_labels"]
        # category_nums = self.check_config["heads"]["category_head"][0]["num_labels"]
        # assert label_data["product_num"] == labels_nums, "商品的数量和我们标签定义的数量不一致，请检查, 实际商品数量：{}，预设标签数量：{}".format(label_data["product_num"],labels_nums)
        # assert label_data["brand_num"] == brand_nums, "品牌的数量和我们标签定义的数量不一致，请检查, 实际品牌数量：{}，预设品牌标签数量：{}".format(label_data["brand_num"],brand_nums)
        # assert label_data["category_num"] == category_nums, "品类的数量和我们标签定义的数量不一致，请检查, 实际品类数量：{}，预设品类标签数量：{}".format(label_data["category_num"],category_nums)
        self.product2id = label_data["product2label"]
        self.id2product = {v:k for k, v in self.product2id.items()}
        self.product2pid = {v:k for k, v in self.product_id2names.items()}
        self.brand2id = label_data["brand2label"]
        self.category2id = label_data["category2label"]
        self.bigcatg2id = label_data["bigcatg2label"]
        self.idx2brand = {idx: brand for brand, idx in self.brand2id.items()}
        self.idx2category = {idx: category for category, idx in self.category2id.items()}
        self.idx2bigcatg = {idx: category for category, idx in self.bigcatg2id.items()}
    def replace_image_local(self, product_image, product_name, check_image=False):
        suffix = product_image.split('.')[-1]
        product_image_folder = os.path.join(self.images_dir, product_name)
        if not os.path.exists(product_image_folder):
            os.makedirs(product_image_folder)
        md5 = self.cal_md5(product_image)  # 图片路径变成md5值
        image_file = os.path.join(product_image_folder, md5 + '.' + suffix)
        assert isinstance(image_file, str), f"图片的路径不是一个字符串，这是有问题的"
        # 下载图片
        download_result, exist_ok = self.download(product_image, image_file, return_exists=True)
        if check_image:
            if not image_loader(image_file):
                print(f"{image_file}加载失败，重新下载")
                download_reuslt, exist_status = self.download(product_image, image_file, return_exists=True, force_download=True)
                # 间隔1秒
                time.sleep(1)
                assert download_reuslt, f"注意：图片经过PIL检查失败，重新下载不成功"
                #再次检查图片，如果失败，返回False
                if not image_loader(image_file):
                    return False, False
        if download_result is False:
            print(f"注意：产品{product_name}的图片下载失败: {product_image}，请检查下载链接是否正确。")
            return False, False
        return image_file,md5
    def read_target_data(self):
        """
        读取所有的商品数据
        :return:
        :rtype:
        """
        data = []
        # 训练样本，每个样本包含【目标商品信息，一个标注，困难样本】
        # 每条包含信息: img: 图像url, text_a: 标题或商品名称, text_b:ocr或别名, "product_name": 商品名称, "label": 商品idx, "brand": 品牌名称, "brand_id"： 品牌idx, "category": 品类名称
        # category_id: 品类idx，
        product2label = self.product2id
        product_name2brand = self.product_name2brand
        product_name2alias = self.product_name2alias
        product_name2picture = self.product_name2picture
        product_name2category = self.product_name2category
        product_name2bigcatg = self.product_name2bigcatg
        bigcatg2label = self.bigcatg2id
        brand2label = self.brand2id
        category2label = self.category2id
        product_id2names = self.product_id2names
        product_name2pid = {v: k for k, v in product_id2names.items()}
        for product, label in tqdm(product2label.items(), desc="进度"):
            print(f"开始进行产品: {product}的处理")
            product_image = product_name2picture.get(product)
            if not product_image:
                print(f"商品: {product} 没有图片链接: {product_image}，跳过")
                continue
            pid = product_name2pid[product]
            # 存在这个商品的标注数据
            product_alias = product_name2alias[product]
            text_b = ";".join(product_alias)
            product_image = product_name2picture.get(product)
            if not product_image:
                print(f"注意：商品的{product}的图片不存在商品图片: {product_image}")
                continue
            md5 = cal_md5(product_image)
            if self.ocr_dict.get(md5):
                text_b = self.ocr_dict.get(md5) + text_b
            target = {
                "img": product_name2picture[product],
                "text_a": product,
                "text_b": text_b,
                "product_name": product,
                "label": label,
                "brand": product_name2brand[product],
                "brand_id": brand2label[product_name2brand[product]],
                "category": product_name2category[product],
                "category_id": category2label[product_name2category[product]],
                "bigcatg": product_name2bigcatg[product],  # 一级品类名称
                "bigcatg_id": bigcatg2label[product_name2bigcatg[product]],  # 一级品类的label
                "pid": pid,
            }
            data.append(target)
        print(f"共包含商品数据: {len(data)} 条")
        return data
    def read_human_data(self):
        """
        读取人工标注数据
        注意，需要数据去重，否则重复输出插入mongo会报错
        :return:
        :rtype:
        """
        df_data = search_and_cache(host="online", db='data_wiz_prod', table="multimodal_entity_linking")
        # 训练样本，每个样本包含【目标商品信息，一个标注，困难样本】
        # 每条包含信息: img: 图像url, text_a: 标题或商品名称, text_b:ocr或别名, "product_name": 商品名称, "label": 商品idx, "brand": 品牌名称, "brand_id"： 品牌idx, "category": 品类名称
        # category_id: 品类idx，
        product2label = self.product2id
        product_name2brand = self.product_name2brand
        product_name2alias = self.product_name2alias
        product_name2picture = self.product_name2picture
        product_name2category = self.product_name2category
        product_name2bigcatg = self.product_name2bigcatg
        bigcatg2label = self.bigcatg2id
        brand2label = self.brand2id
        category2label = self.category2id
        product_id2names = self.product_id2names
        product_name2pid = {v: k for k, v in product_id2names.items()}
        data = []
        unique_md5 = []
        repeat_cnt = 0 #重复标注数据
        for row_index, row in df_data.iterrows():
            status = row["status"]
            # 只要人工确认的
            if status != 2:
                continue
            pid = row["pid"]
            pid_str = str(int(pid))
            if pid_str not in product_id2names:
                continue
            product = product_id2names[pid_str]
            url_id = row["url_id"]
            title = row["title"]
            brand_name = product_name2brand[product]
            img = row["pic"]
            md5 = cal_md5(img)
            text_b = self.ocr_dict.get(md5, "")
            # 利用多个字段的组合去重
            unimd5 = cal_md5(content=f"{img}{url_id}{title}{product}")
            if unimd5 in unique_md5:
                repeat_cnt += 1
                continue
            else:
                unique_md5.append(unimd5)
            page_info = {
                "img": img,
                "text_a": title,
                "text_b": text_b,
                "product_name": product,
                "label": product2label[product],
                "brand": brand_name,
                "brand_id": brand2label[product_name2brand[product]],
                "category": product_name2category[product],
                "category_id": category2label[product_name2category[product]],
                "bigcatg": product_name2bigcatg[product],  # 一级品类名称
                "bigcatg_id": bigcatg2label[product_name2bigcatg[product]],  # 一级品类的label
                "pid": pid,
            }
            data.append(page_info)
        assert len(data) != 0, "没有读取人工标注的数据"
        print(f"共包含商品数据: {len(data)} 条，共包含重复数据: {repeat_cnt} 条")
        return data
    def build_product_encoder(self, mini=False):
        """
        所有商品生成向量
        1、从线上mysql读取数据
        2、生成向量
        3、保存向量
        Args:
        """
        print(f"所有商品数据生成向量")
        data = self.read_target_data()
        process_data = []
        if mini:
            data = data[:100]
        for one in data:
            one["img_url"] = one["img"]
            result, md5 = self.replace_image_local(product_image=one["img"], product_name=one["product_name"])
            if result is False:
                continue
            one["image"] = result
            one["md5"] = md5
            # 对于text_b, 是从图片中进行OCR识别的结果
            process_data.append(one)
        # 处理下图片数据，如果给定的data中的图片是http的链接，那么首先查看是否已缓存，如果已缓存，直接返回缓存路径，否则下载后，返回缓存路径
        cls_vectors = self.encoder(process_data)
        # 把原始文本也放回去
        assert len(data) == len(cls_vectors), f"模型预测结束后，商品的数量和预测的结果的数量不一致"
        self.save_vectors(cls_vectors=cls_vectors, products=data, collection=f"product_vectors_{self.time_key}", clean_before_insert=True)
    def build_human_encoder(self, start_batch=0, check_image=False):
        """
        已标注数据生成向量
        1、从线上mysql读取数据
        2、生成向量
        3、保存向量
        Args:
        """
        batch_size = 100
        print(f"所有标注数据生成向量")
        data = self.read_human_data()
        if start_batch != 0:
            data = data[batch_size*start_batch:]
        process_data = []
        for one in tqdm(data, desc="图片路径检查中"):
            one["img_url"] = one["img"]
            result, md5 = self.replace_image_local(product_image=one["img"], product_name=one["product_name"], check_image=check_image)
            if result is False:
                continue
            one["image"] = result
            one["md5"] = md5
            # 对于text_b, 是从图片中进行OCR识别的结果
            process_data.append(one)
        print(f"处理前总数据量: {len(data)}, 处理完成后，数据剩余: {len(process_data)} 条")
        total_batch = len(process_data) // batch_size + (1 if len(process_data) % batch_size > 0 else 0)
        total_batch_list = list(range(total_batch))
        for idx in tqdm(total_batch_list, desc="生成进度"):
            sub_list = process_data[idx * batch_size:(idx + 1) * batch_size]
            # 处理下图片数据，如果给定的data中的图片是http的链接，那么首先查看是否已缓存，如果已缓存，直接返回缓存路径，否则下载后，返回缓存路径
            cls_vectors = self.encoder(sub_list)
            # 把原始文本也放回去
            assert len(sub_list) == len(cls_vectors), f"模型预测结束后，数量和预测的结果的数量不一致"
            if idx == 0 and start_batch == 0:
                clean_before_insert = True
            else:
                clean_before_insert = False
            self.save_vectors(cls_vectors=cls_vectors, products=sub_list, collection=f"human_vectors_{self.time_key}",clean_before_insert=clean_before_insert)

    def save_vectors(self, cls_vectors, products, collection, clean_before_insert=True):
        """
        保存向量到mongo数据库中
        :return:
        :rtype:
        """
        data = []
        for cls, product in zip(cls_vectors, products):
            if "build" in product:
                # 不要build关键字
                product = product["build"]
            assert cls["instance_id"] == product["md5"],f"模型在推理数据时造成了混乱，请检查数据: {product}"
            cls_list = cls["vector"]
            product["cls_vector"] = cls_list
            data.append(product)
        mongo_insert(data=data, database='label', collection=collection, clean_before_insert=clean_before_insert)
        print(f"保存到collection： {collection}完成")

def mongo_insert(data, database='label', collection='tmall', clean_before_insert=False, only_clean=False):
    """
    插入数据到mongo中
    :param data: list
    :type data:
    :param clean_before_insert: 在插入之前，先清空数据
    :param only_clean: 只做清空操作，不做插入
    :return:
    :rtype:
    """
    mongo_host = "192.168.50.189"
    client = pymongo.MongoClient(mongo_host, 27017)
    db = client[database]
    # 选择哪个collections
    mycol = db[collection]
    #插入数据
    if clean_before_insert or only_clean:
        x = mycol.delete_many({})
        print(f"事先清除已有数据成功: 清除的collection是: {database}中的{collection}")
    if only_clean:
        print(f"清空完成")
    else:
        x = mycol.insert_many(data)
        print(f"插入成功，插入的id是{x}")
def mongo_read(database='label', collection='labeled_data'):
    """
    返回database中collection的所有数据,
    last_one: 默认返回最后一条数据
    :param database:
    :type database:
    :param collection:
    :type collection:
    :return:
    :rtype:
    """
    mongo_host = "192.168.50.189"
    client = pymongo.MongoClient(mongo_host, 27017)
    db = client[database]
    # 选择哪个collections
    mycol = db[collection]
    data = []
    for x in mycol.find():
        data.append(x)
    logging.info(f"从mongo数据库{collection}中共搜索到所有数据{len(data)}条")
    return data
def mongo_find(query, database='label', collection='tmall'):
    """
    返回database中collection的所有数据
    :params: query: 搜索 eg: { "time_key": "20221111094612" }
    :param database:
    :type database:
    :param collection:
    :type collection:
    :return:
    :rtype:
    """
    client = pymongo.MongoClient("192.168.50.189", 27017)
    db = client[database]
    # 选择哪个collections
    mycol = db[collection]
    data = []
    for x in mycol.find(query):
        data.append(x)
    print(f"从mongo数据库{collection}中共搜索到所有数据{len(data)}条")
    return data
def search_and_cache(host,db,table=None,sql=None,flush_cache=False, flush_cache_bytime=True, use_cache=True, user=None, port=None,password=None, verbose=True):
    """
    :param host: mysql的host
    :type host:
    :param db: mysql的database
    :type db:
    :param sql: sql语句, eg: select * from school limit 3， 在线查sql，返回sql的内容
    :type sql: 如果给定sql，那么不用给定table，如果给定table字段，不用给定sql
    :param table: 某个表，用于缓存用，缓存整个table内容
    :param flush_cache: 是否从线上拉下来sql数据，缓存到本地
    :type flush_cache:
    :param flush_cache_bytime: 根据时间来决定是否进行更新cache，当cache的时间超过24小时候，选择更新
    :param use_cache: 是否使用本地的缓存的sql数据，还是从线上获取
    :type use_cache:
    :param user: 使用的mysql用户
    :type user:
    :param port: 使用的mysql端口
    :type port:
    :param password: 使用的mysql密码
    :return: dataframe的格式
    :rtype:
    """
    #缓存sql的路径
    cache_path = "/Users/admin/Documents/lavector"
    if not os.path.exists(cache_path):
        cache_path = "./"
    hosts_info = {
        "129": {"host": "192.168.50.129", "port": 3306, "user": "test", "password": "123456"},
        "online": {"host": "lavector-mysql.cqksigp8aiow.rds.cn-northwest-1.amazonaws.com.cn", "port": 3306, "user": "lavector", "password": "passw0rd"},
    }
    # 验证给的host是否在hosts_info中
    host_info = hosts_info.get(host)
    if not host_info:
        # 如果没找到mysql的预留信息，那么必须给定user，password等信息
        assert user and port and password, "给定的host没有找到对应的内置数据库信息，而且user，port，password也没有给定，请检查"
    mysql_host = host_info["host"]
    mysql_user = host_info["user"]
    mysql_port = host_info["port"]
    mysql_password = host_info["password"]
    if sql:
        if verbose:
            print(f"通过sql查询")
        # 检查是否有cache
        sql_string = sql.replace('"','').replace(' ','_').replace('*','').replace('=','').replace('-','')
        cache_name = f"{host}_{db}_{table}_{sql_string}.sql"
        full_path = os.path.join(cache_path, cache_name)
        if os.path.exists(full_path):
            file_create_time = os.stat(full_path).st_ctime
        else:
            file_create_time = 0
        # 或者文件创建时间大于24小时，那么自动更新
        flush_bytime = False
        if flush_cache_bytime:
            if file_create_time < time.time()-24*60*60:
                flush_bytime = True
        if flush_cache or (use_cache and not os.path.exists(full_path)) or flush_bytime:
            if verbose:
                print(f"缓存的mysql table 文件不存在或者更新缓存,或超时更新: {full_path}，将会重新获取")
                #更新缓存文件
                #连接数据库
            pydb = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=db, port=mysql_port)
            df = pd.read_sql(sql, pydb)
            df.to_pickle(full_path)
        else:
            if verbose:
                print(f"读取缓存的整个table内容: {full_path}")
            df = pd.read_pickle(full_path)
    else:
        # 检查是否有cache
        cache_name = f"{host}_{db}_{table}.sql"
        full_path = os.path.join(cache_path, cache_name)
        if os.path.exists(full_path):
            file_create_time = os.stat(full_path).st_ctime
        else:
            file_create_time = 0
        # 或者文件创建时间大于24小时，那么自动更新
        flush_bytime = False
        if flush_cache_bytime:
            if file_create_time < time.time()-24*60*60:
                flush_bytime = True
        if flush_cache or not use_cache or flush_bytime or not os.path.exists(full_path):
            if verbose:
                print(f"缓存的mysql table 文件不存在或者更新缓存,或超时更新: {full_path}，将会重新获取")
            #更新缓存文件
            #连接数据库
            pydb = pymysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=db, port=mysql_port)
            cache_sql = f'select * from {table}'
            df = pd.read_sql(cache_sql, pydb)
            df.to_pickle(full_path)
        else:
            if verbose:
                print(f"读取缓存的整个table内容: {full_path}")
            df = pd.read_pickle(full_path)
    # 开始返回sql的查询结果
    if verbose:
        print(f"此次SQL查询返回数据条数{len(df)}")
    return df
def cal_md5(content):
    """
    计算content字符串的md5
    :param content:
    :return:
    """
    # 使用encode
    result = hashlib.md5(content.encode())
    # 打印hash
    md5 = result.hexdigest()
    return md5

def image_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            img_rgb = img.convert("RGB")
            return True
    except Exception as e:
        print("加载图片失败:", path)
        print(e)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态预测api")
    parser.add_argument("-k","--time_key", type=str,required=True, help="使用数据库中的哪个timekey")
    parser.add_argument("-c","--checkpoint", type=str,help="使用哪个checkpoint")
    parser.add_argument("-cfg","--cfg-path", type=str,default="lavis/projects/blip2/eval/cosmetic_ft_eval.yaml",help="配置文件")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    if not args.checkpoint:
        checkpoint_path = f"lavis/output/checkpoint_{args.time_key}.pth"
        assert os.path.exists(checkpoint_path), f"模型路径: {checkpoint_path} 不存在，请检查"
        args.checkpoint = checkpoint_path
    inference = Inference(time_key=args.time_key, checkpoint_path=args.checkpoint)
    inference.build_human_encoder(start_batch=0, check_image=True)
    inference.build_product_encoder(mini=False)
