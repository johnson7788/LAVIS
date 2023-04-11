"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import logging

import numpy as np
import torch
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("multimodal_retrieval")
class MultimodalRetrievalTask(BaseTask):
    def __init__(self):
        super().__init__()

    def valid_step(self, model, samples):
        results = []

        outputs = model.predict(samples)

        predictions = outputs["predictions"]
        targets = outputs["targets"]
        vectors = outputs["vectors"] # 向量
        brand_logits = outputs["brand_logits"] # 向量
        category_logits = outputs["category_logits"] # 向量
        bigcatg_logits = outputs["bigcatg_logits"] # 向量

        predictions_logits = predictions.max(1)[1].cpu().numpy()
        scores = predictions.cpu().numpy()
        # brand_predictions = brand_logits.max(1)[1].cpu().numpy()
        # category_predictions = category_logits.max(1)[1].cpu().numpy()
        # bigcatg_predictions = bigcatg_logits.max(1)[1].cpu().numpy()
        brand_logits = brand_logits.cpu().numpy()
        category_logits = category_logits.cpu().numpy()
        bigcatg_logits = bigcatg_logits.cpu().numpy()
        targets = targets.cpu().numpy()
        vectors = vectors.cpu().numpy()

        indices = samples[self.inst_id_key]

        for pred, score, tgt, index, vector, brand, category, bigcatg in zip(predictions_logits, scores, targets, indices, vectors,brand_logits,category_logits,bigcatg_logits):
            if isinstance(index, torch.Tensor):
                index = index.item()

            results.append(
                {
                    self.inst_id_key: index,
                    "prediction": pred.item(),
                    "target": tgt.item(),
                    "scores": score,
                    "vector": vector,
                    "brand_scores": brand,
                    "category_scores": category,
                    "bigcatg_scores": bigcatg,
                }
            )

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        # val_result只保留prediction,target和instance_id
        new_val_result = []
        for res in val_result:
            new_val_result.append(
                {
                    self.inst_id_key: res[self.inst_id_key],
                    "prediction": res["prediction"],
                    "target": res["target"],
                }
            )
        eval_result_file = self.save_result(
            result=new_val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=self.inst_id_key,
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        results = json.load(open(eval_result_file))

        predictions = np.array([res["prediction"] for res in results])
        targets = np.array([res["target"] for res in results])

        accuracy = (targets == predictions).sum() / targets.shape[0]
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics
