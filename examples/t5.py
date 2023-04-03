#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/4/3 17:34
# @File  : t5.py
# @Author: 
# @Desc  :


import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, feature_dim, num_classes, margin=0.5, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        # Normalize features and weights
        features_norm = F.normalize(features, dim=1)
        weights_norm = F.normalize(self.weight, dim=1)

        # Calculate cos(theta)
        cos_theta = torch.matmul(features_norm, weights_norm.t())

        # Add margin to the target logits
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        cos_theta_m = cos_theta - one_hot * self.margin

        # Calculate the final logits
        logits = cos_theta_m * self.scale

        # Calculate ArcFace loss
        loss = F.cross_entropy(logits, labels)
        return loss

def main():
    # Parameters
    batch_size = 16
    feature_dim = 512
    num_classes = 10

    # Generate random tensors as inputs and outputs
    features = torch.rand(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))  #tensor([1, 4, 2, 0, 2, 7, 9, 3, 3, 6, 4, 6, 4, 4, 4, 1])

    # Compute the ArcFace loss
    arcface_loss = ArcFaceLoss(feature_dim, num_classes)
    loss = arcface_loss(features, labels)
    print(f"ArcFace loss: {loss.item()}")

if __name__ == "__main__":
    main()