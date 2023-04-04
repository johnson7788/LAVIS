"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class ComesticDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # 加载图片的id
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["md5"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        caption_text = ann["text_a"] + " " + ann["text_b"]
        image = self.vis_processor(image)
        caption = self.text_processor(caption_text)
        label = ann["label"]
        brand_id = ann["brand_id"]
        category_id = ann['category_id']
        bigcatg_id = ann['bigcatg_id']
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["md5"]],
            "label": label,
            "brand_id": brand_id,
            "category_id": category_id,
            "bigcatg_id": bigcatg_id,
        }


class ComesticEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        caption_text = ann["text_a"] + " " + ann["text_b"]
        caption = self.text_processor(caption_text)
        label = ann["label"]
        brand_id = ann["brand_id"]
        category_id = ann['category_id']
        bigcatg_id = ann['bigcatg_id']
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["md5"]],
            "label": label,
            "brand_id": brand_id,
            "category_id": category_id,
            "bigcatg_id": bigcatg_id,
            "instance_id": ann["md5"],
        }
