import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import nn
from torch.nn import functional as F

from multimil.nn import MLP, Aggregator
from multimil.utils import prep_minibatch, select_covariates

class UnsupervisedMILTorch(BaseModuleClass):
    def __init__(
        self,
        z_dim=16,
        dropout=0.2,
        normalization="layer",
        scoring="gated_attn",
        attn_dim=16,
        n_layers_cell_aggregator=1,
        n_layers_mlp_attn=1,
        n_hidden_cell_aggregator=16,
        n_hidden_mlp_attn=16,
        clustering_loss_coef=1.0,   # Weighting for the clustering loss.
        n_clusters=10,              # Number of clusters to use.
        tau=1.0,                    # Temperature-like scaling factor.
        sample_batch_size=32,       # Lowered to generate more bag embeddings per training batch.
        activation="leaky_relu",
        initialization=None,
    ):
        super().__init__()

        # Choose activation function.
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "relu":
            self.activation = nn.ReLU
        else:
            raise NotImplementedError(
                f'activation must be one of ["leaky_relu","tanh","relu"], but got {activation}.'
            )

        self.clustering_loss_coef = clustering_loss_coef
        self.n_clusters = n_clusters
        self.tau = tau
        self.sample_batch_size = sample_batch_size

        # Build cell-level aggregator: an MLP followed by an Aggregator.
        self.cell_level_aggregator = nn.Sequential(
            MLP(
                n_input=z_dim,
                n_output=z_dim,
                n_layers=n_layers_cell_aggregator,
                n_hidden=n_hidden_cell_aggregator,
                dropout_rate=dropout,
                activation=self.activation,
                normalization=normalization,
            ),
            Aggregator(
                n_input=z_dim,
                scoring=scoring,
                attn_dim=attn_dim,
                sample_batch_size=sample_batch_size,
                scale=True,
                dropout=dropout,
                n_layers_mlp_attn=n_layers_mlp_attn,
                n_hidden_mlp_attn=n_hidden_mlp_attn,
                activation=self.activation,
            ),
        )

        # If clustering is enabled, learn cluster centers.
        if self.n_clusters > 0 and self.clustering_loss_coef > 0:
            self.mu = nn.Parameter(torch.randn(self.n_clusters, z_dim))
        else:
            self.mu = None

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        return {"x": x}

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        return {"z": z}

    @auto_move_data
    def inference(self, x) -> dict:
        """
        Inference:
          - Splits cell embeddings into contiguous bags of size `sample_batch_size`.
          - Aggregates each bag via the cell_level_aggregator.
        """
        z = x
        inference_outputs = {"z": z}
        batch_size = x.shape[0]
        # Partition x into bags.
        idx = list(range(self.sample_batch_size, batch_size, self.sample_batch_size))
        if batch_size % self.sample_batch_size != 0:
            idx = []  # For simplicity, skip partial bags.
        zs = torch.tensor_split(z, idx, dim=0)  # List of [bag_size, z_dim]
        zs = torch.stack(zs, dim=0)  # Shape: (num_bags, bag_size, z_dim)
        zs_attn = self.cell_level_aggregator(zs)  # Aggregated bag embeddings, shape: (num_bags, z_dim)
        inference_outputs["bag_embeddings"] = zs_attn
        return inference_outputs

    @auto_move_data
    def generative(self, z) -> torch.Tensor:
        return z  # Placeholder.

    def compute_clustering_loss(self, z_bag: torch.Tensor) -> torch.Tensor:
        """
        Compute a k-means–style clustering loss on bag embeddings.
        """
        # Compute squared Euclidean distances (scaled by tau) between each bag embedding and cluster centers.
        dist = self.tau * torch.sum((z_bag.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2, dim=2)
        # Subtract row-mean for numerical stability.
        mean_dist = torch.mean(dist, dim=1, keepdim=True)
        temp_dist = dist - mean_dist
        # Soft assignment.
        q = torch.exp(-temp_dist)
        q = q / torch.sum(q, dim=1, keepdim=True)
        # Sharpen the assignments.
        q = q ** 2
        q = q / torch.sum(q, dim=1, keepdim=True)
        # Loss is the weighted average distance.
        loss_cluster = torch.mean(torch.sum(dist * q, dim=1))
        return loss_cluster

    def _calculate_loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        loss = torch.tensor(0.0, device=self.device)
        extra_metrics = {}

        # Add clustering loss if there are at least two bag embeddings.
        z_bag = inference_outputs.get("bag_embeddings", None)
        if self.mu is not None and z_bag is not None and z_bag.shape[0] > 1:
            clustering_loss = self.compute_clustering_loss(z_bag)
            loss += self.clustering_loss_coef * clustering_loss
            extra_metrics["clustering_loss"] = clustering_loss

        # Placeholders for recon_loss and kl_loss.
        minibatch_size = tensors[REGISTRY_KEYS.X_KEY].shape[0]
        recon_loss = torch.zeros(minibatch_size, device=self.device)
        kl_loss = torch.zeros(minibatch_size, device=self.device)
        return loss, recon_loss, kl_loss, extra_metrics

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        loss, recon_loss, kl_loss, extra_metrics = self._calculate_loss(
            tensors, inference_outputs, generative_outputs, kl_weight
        )
        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_loss,
            extra_metrics=extra_metrics,
        )

    def select_losses_to_plot(self):
        loss_names = []
        if self.n_clusters > 0 and self.clustering_loss_coef != 0:
            loss_names.append("clustering_loss")
        return loss_names
