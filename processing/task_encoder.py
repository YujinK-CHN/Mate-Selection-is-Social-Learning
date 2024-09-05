# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Component to encode the task."""

import json

import torch
import torch.nn as nn


class TaskEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        """Encode the task into a vector.

        Args:
            pretrained_embedding_cfg (ConfigType): config for using pretrained
                embeddings.
            num_embeddings (int): number of elements in the embedding table. This is
                used if pretrained embedding is not used.
            embedding_dim (int): dimension for the embedding. This is
                used if pretrained embedding is not used.
            hidden_dim (int): dimension of the hidden layer of the trunk.
            num_layers (int): number of layers in the trunk.
            output_dim (int): output dimension of the task encoder.
        """
        super(TaskEncoder, self).__init__()
        self.embedding = nn.Sequential(
                nn.Embedding(num_embeddings = num_embeddings, embedding_dim = embedding_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, env_index: torch.Tensor) -> torch.Tensor:
        return self.embedding(env_index)
