"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import T5TokenizerFast
from lavis.models.blip2_models.modeling_t5 import T5Config, T5EncoderModel
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig,OPTModel
from lavis.models.blip2_models.face_head import build_head
from transformers import AutoTokenizer


@registry.register_model("blip2_lora")
class Blip2Lora(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_lora", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        text_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        loss_head_type="adaface",
        class_num=70722,
        brand_num=800,
        category_num=161,
        bigcatg_num=4,
    ):
        super().__init__()
        self.class_num = class_num
        self.tokenizer = self.init_tokenizer()
        # 初始化视觉编码器，例如eva_clip_g
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("冻结视觉编码器")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.head = build_head(head_type=loss_head_type,embedding_size=self.Qformer.config.hidden_size,class_num=class_num,m=0.4,h=0.333,s=64.,t_alpha=1.0)
        self.brand_head = build_head(head_type=loss_head_type,embedding_size=self.Qformer.config.hidden_size,class_num=brand_num,m=0.4,h=0.333,s=64.,t_alpha=1.0)
        self.category_head = build_head(head_type=loss_head_type,embedding_size=self.Qformer.config.hidden_size,class_num=category_num,m=0.4,h=0.333,s=64.,t_alpha=1.0)
        self.bigcatg_head = build_head(head_type=loss_head_type,embedding_size=self.Qformer.config.hidden_size,class_num=bigcatg_num,m=0.4,h=0.333,s=64.,t_alpha=1.0)
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
        # 注意，不同的text_model，使用不同的tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        text_config = OPTConfig.from_pretrained(text_model)
        self.text_model = OPTModel.from_pretrained(
            text_model, config=text_config
        )
        for name, param in self.text_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()
        # 图像特征处理
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size)
        # 文本特征处理
        self.text_proj = nn.Linear(
            self.text_model.config.hidden_size, self.Qformer.config.hidden_size
        )
        # 品牌的预测
        self.brand_proj = nn.Linear(self.Qformer.config.hidden_size, brand_num)
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        self.cross_entropy_loss = CrossEntropyLoss()

    def forward(self, samples):
        image = samples["image"]   # [batch_size, 3, 364, 364] # 获取输入数据中的图像
        caption = samples["text_input"]
        labels = samples["label"]
        brand_id = samples["brand_id"]
        category_id = samples["category_id"]
        bigcatg_id = samples["bigcatg_id"]
        # 使用VIT处理图像特征处理-------------------------
        with self.maybe_autocast():  #处理图像特征,自动混合精度
            image_embeds = self.ln_vision(self.visual_encoder(image)) # 图像特征经过视觉编码器和层归一化处理后得到的嵌入特征，[batch_size, 677, 1408]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        ) #图像特征的注意力掩码 【batch_size, 677】
        # 处理图像特征
        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # 查询语句的嵌入特征，[batch_size, 32, 768]
        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     return_dict=True,
        # )
        #query_output dict, 使用Qformer模型对查询语句和图像特征进行编码, last_hidden_state: [batch_size, 32, 768]
        # inputs_opt图像的特征, image_features:[batch_size, 32, 256]
        # image_features = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state), dim=-1
        # )
        # 使用t5模型处理文本特征处理-------------------------
        with self.maybe_autocast(dtype=torch.bfloat16):
            text = self.text_tokenizer(
                caption,
                truncation=True,
                padding="longest",
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            text_output = self.text_model(
                input_ids=text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )

        text_features = F.normalize(
            self.text_proj(text_output.last_hidden_state), dim=-1
        )
        # text_features = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        # 多模态特征处理,使用Qformer处理-------------------------
        # 图像和文本的掩码
        # attention_mask = torch.cat([image_atts, text.attention_mask], dim=1)
        # attention_mask = text.attention_mask
        # image_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask = torch.cat([image_mask, text.attention_mask], dim=1)
        # query_embeds = torch.cat([query_tokens, text_embeds], dim=2)
        # 图像和文本的嵌入特征，Qformer使用的"bert-base-uncased"初始化，如果直接传入input_ids，有中文，可能会有问题
        output = self.Qformer.bert(
            query_embeds=text_features, #图像特征
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        # 多模态的嵌入特征
        multimodal_embeds = output.last_hidden_state
        # 取第一个token的嵌入特征
        multimodal_embeds_first = multimodal_embeds[:, 0, :]
        # L2范数, 1: 指定 norm 运算会在 multimodal_embeds 的第二维度(index=1)上进行。也可以选择 0 进行行wise norm,或者更高维等。
        # True: 选择是否返回 norm 后的值,True 时 norm 会返回范数值,False 时仅进行范数计算但不返回值。
        norm = torch.norm(multimodal_embeds_first, 2, 1, True)
        multimodal_embeds_output = multimodal_embeds_first.div(norm)
        cos_thetas = self.head(embbedings=multimodal_embeds_output, norms=norm, label=labels)
        # 产品的损失
        loss_product = self.cross_entropy_loss(cos_thetas, labels)
        # 品牌的损失
        cos_brand = self.brand_head(embbedings=multimodal_embeds_output, norms=norm, label=brand_id)
        loss_brand = self.cross_entropy_loss(cos_brand, brand_id)
        # 类别的损失
        cos_category = self.category_head(embbedings=multimodal_embeds_output, norms=norm, label=category_id)
        loss_category = self.cross_entropy_loss(cos_category, category_id)
        # 大品类的损失
        cos_bigcatg = self.bigcatg_head(embbedings=multimodal_embeds_output, norms=norm, label=bigcatg_id)
        loss_bigcatg = self.cross_entropy_loss(cos_bigcatg, bigcatg_id)
        # 品牌直接预测的损失
        brand_logits = self.brand_proj(multimodal_embeds_output)
        loss_brand_direct = self.cross_entropy_loss(brand_logits, brand_id)
        # 总的损失
        loss = loss_product + loss_brand + loss_category + loss_brand_direct + loss_bigcatg
        logging.info("loss_product: {}, loss_brand: {}, loss_category: {}, loss_brand_direct: {}, loss_bigcatg: {}".format(loss_product, loss_brand, loss_category, loss_brand_direct, loss_bigcatg))
        logging.info("total_loss: {}".format(loss))
        # 计算损失
        return {"loss": loss}

    @torch.no_grad()
    def predict(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # bs = samples["image"].shape[0]
        # fake_result = {
        #     "predictions": torch.randn(bs, 256),
        #     "targets": torch.randn(bs),
        # }
        # return fake_result # TODO,临时返回
        image = samples["image"]   # [batch_size, 3, 364, 364] # 获取输入数据中的图像
        caption = samples["text_input"]
        labels = samples["label"]
        # 使用VIT处理图像特征处理-------------------------
        with self.maybe_autocast():  #处理图像特征,自动混合精度
            image_embeds = self.ln_vision(self.visual_encoder(image)) # 图像特征经过视觉编码器和层归一化处理后得到的嵌入特征，[batch_size, 677, 1408]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        ) #图像特征的注意力掩码 【batch_size, 677】
        # 处理图像特征
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # 查询语句的嵌入特征，[batch_size, 32, 768]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ) #query_output dict, 使用Qformer模型对查询语句和图像特征进行编码, last_hidden_state: [batch_size, 32, 768]
        # inputs_opt图像的特征, image_features:[batch_size, 32, 256]
        image_features = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )
        # 使用t5模型处理文本特征处理-------------------------
        with self.maybe_autocast(dtype=torch.bfloat16):
            text = self.text_tokenizer(
                caption,
                truncation=True,
                padding="longest",
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            text_output = self.text_model(
                input_ids=text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )

        text_features = F.normalize(
            self.text_proj(text_output.last_hidden_state), dim=-1
        )
        # text_features = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        # 多模态特征处理,使用Qformer处理-------------------------
        # 图像和文本的掩码
        # attention_mask = torch.cat([image_atts, text.attention_mask], dim=1)
        # attention_mask = text.attention_mask
        image_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # attention_mask = torch.cat([image_mask, text.attention_mask], dim=1)
        # query_embeds = torch.cat([query_tokens, text_embeds], dim=2)
        # 图像和文本的嵌入特征，Qformer使用的"bert-base-uncased"初始化，如果直接传入input_ids，有中文，可能会有问题
        output = self.Qformer.bert(
            query_embeds=text_features, #图像特征
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        # 多模态的嵌入特征
        multimodal_embeds = output.last_hidden_state
        # 取第一个token的嵌入特征
        multimodal_embeds_first = multimodal_embeds[:, 0, :]
        # L2范数, 1: 指定 norm 运算会在 multimodal_embeds 的第二维度(index=1)上进行。也可以选择 0 进行行wise norm,或者更高维等。
        # True: 选择是否返回 norm 后的值,True 时 norm 会返回范数值,False 时仅进行范数计算但不返回值。
        norm = torch.norm(multimodal_embeds_first, 2, 1, True)
        multimodal_embeds_output = multimodal_embeds_first.div(norm)
        logits = torch.mm(multimodal_embeds_output, self.head.kernel)
        category_logits = torch.mm(multimodal_embeds_output, self.category_head.kernel)
        bigcatg_logits = torch.mm(multimodal_embeds_output, self.bigcatg_head.kernel)
        brand_logits = self.brand_proj(multimodal_embeds_output)

        # 返回向量就行了，对于评估来说
        result = {
            "predictions": logits,
            "targets": labels,
            "vectors": multimodal_embeds_output,
            "brand_logits": brand_logits,
            "category_logits": category_logits,
            "bigcatg_logits": bigcatg_logits,
        }
        return result

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        # t5_model = cfg.get("t5_model")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        class_num = cfg.get("class_num")
        loss_head_type = cfg.get("loss_head_type")

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            text_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            loss_head_type=loss_head_type,
            class_num=class_num,
            brand_num=cfg.get("brand_num"),
            category_num=cfg.get("category_num"),
            bigcatg_num=cfg.get("bigcatg_num"),
        )
        model.load_checkpoint_from_config(cfg)

        return model
