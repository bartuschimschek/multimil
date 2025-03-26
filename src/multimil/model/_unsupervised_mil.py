import logging
from typing import Union

import torch
from anndata import AnnData
from pytorch_lightning.callbacks import ModelCheckpoint
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.model._utils import parse_device_args
from scvi.model.base import ArchesMixin, BaseModelClass
from scvi.model.base._archesmixin import _get_loaded_data
from scvi.model.base._utils import _initialize_model
from scvi.train import AdversarialTrainingPlan, TrainRunner
from scvi.train._callbacks import SaveBestState

from multimil.dataloaders import GroupAnnDataLoader, GroupDataSplitter
from multimil.module import UnsupervisedMILTorch
from multimil.utils import plt_plot_losses

class UnsupervisedMIL(BaseModelClass, ArchesMixin):
    """
    An unsupervised Multi-instance Learning (MIL) model that aggregates cell-level embeddings into bag-level embeddings without relying on disease labels.
    The model optionally employs clustering-based loss functions applied directly to bag embeddings to encourage meaningful representation learning.
    """

    def __init__(
        self,
        adata,
        sample_key: str,
        sample_batch_size: int = 256,
        bag_size: int = 32,
        normalization: str = "layer",
        z_dim: int = 16,
        dropout: float = 0.2,
        scoring: str = "gated_attn",
        attn_dim: int = 16,
        n_layers_cell_aggregator: int = 1,
        n_layers_mlp_attn: int = 1,
        n_hidden_cell_aggregator: int = 128,
        n_hidden_mlp_attn: int = 32,
        activation: str = "leaky_relu",
        initialization: Union[str, None] = None,
        ignore_covariates = None,
        clustering_loss_coef: float = 1.0,
        n_clusters: int = 10,
        tau: float = 1.0,
    ):
        super().__init__(adata)
        self.sample_key = sample_key
        # Ensure the sample key is among the registered categorical covariates.
        if self.sample_key not in self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
            raise ValueError(
                f"Sample key '{self.sample_key}' must be one of the registered categorical covariates: "
                f"{self.adata_manager.registry['setup_args']['categorical_covariate_keys']}"
            )
        # Instantiate the unsupervised MIL module.
        self.module = UnsupervisedMILTorch(
            z_dim=z_dim,
            dropout=dropout,
            normalization=normalization,
            scoring=scoring,
            attn_dim=attn_dim,
            n_layers_cell_aggregator=n_layers_cell_aggregator,
            n_layers_mlp_attn=n_layers_mlp_attn,
            n_hidden_cell_aggregator=n_hidden_cell_aggregator,
            n_hidden_mlp_attn=n_hidden_mlp_attn,
            activation=activation,
            initialization=initialization,
            clustering_loss_coef=clustering_loss_coef,
            n_clusters=n_clusters,
            tau=tau,
            sample_batch_size=bag_size,  # bag_size is used to form the bags.
        )
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int = 200,
        lr: float = 5e-4,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        train_size: float = 0.9,
        validation_size: Union[float, None] = None,
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        eps: float = 1e-8,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        n_steps_kl_warmup: Union[int, None] = None,
        adversarial_mixing: bool = False,
        plan_kwargs: dict = None,
        early_stopping_monitor: Union[str, None] = "clustering_loss_validation",
        early_stopping_mode: Union[str, None] = "min",
        save_checkpoint_every_n_epochs: Union[int, None] = None,
        path_to_checkpoints: Union[str, None] = None,
        **kwargs,
    ):
        if n_epochs_kl_warmup is None:
            n_epochs_kl_warmup = max(max_epochs // 3, 1)
        update_dict = {
            "lr": lr,
            "adversarial_classifier": adversarial_mixing,
            "weight_decay": weight_decay,
            "eps": eps,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            "optimizer": "AdamW",
            "scale_adversarial_loss": 1,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if save_best:
            if "callbacks" not in kwargs:
                kwargs["callbacks"] = []
            kwargs["callbacks"].append(SaveBestState(monitor=early_stopping_monitor, mode=early_stopping_mode))
        if save_checkpoint_every_n_epochs is not None:
            if path_to_checkpoints is not None:
                kwargs["callbacks"].append(
                    ModelCheckpoint(
                        dirpath=path_to_checkpoints,
                        save_top_k=-1,
                        monitor="epoch",
                        every_n_epochs=save_checkpoint_every_n_epochs,
                        verbose=True,
                    )
                )
            else:
                raise ValueError(
                    f"`save_checkpoint_every_n_epochs={save_checkpoint_every_n_epochs}` requires `path_to_checkpoints`."
                )
        data_splitter = GroupDataSplitter(
            self.adata_manager,
            group_column=self.sample_key,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
        training_plan = AdversarialTrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_mode=early_stopping_mode,
            early_stopping_patience=50,
            enable_checkpointing=True,
            **kwargs,
        )
        return runner()

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        categorical_covariate_keys: list[str] = None,
        continuous_covariate_keys: list[str] = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer=None),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, None),
            fields.CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_model_output(
        self,
        adata: AnnData = None,
        batch_size: int = 256,
    ):
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata,
            batch_size=batch_size,
            min_size_per_class=batch_size,
            data_loader_class=GroupAnnDataLoader,
            shuffle=False,
            shuffle_classes=False,
            group_column=self.sample_key,
            drop_last=False,
        )

        bag_embeddings = []
        cell_level_attn = []

        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            # Collect bag-level embeddings
            bag_embeddings.append(outputs["bag_embeddings"].cpu())

            # Collect cell-level attention weights
            cell_attn = self.module.cell_level_aggregator[-1].A.squeeze(dim=1)
            cell_attn = cell_attn.flatten()  # shape: number_of_bags * bag_size
            cell_level_attn.append(cell_attn.cpu())

        # Store bag embeddings in .uns, no shape constraints
        if bag_embeddings:
            adata.uns["bag_embeddings"] = torch.cat(bag_embeddings).numpy()

        # Store cell-level attention in .obs["cell_attn"]
        if cell_level_attn:
            adata.obs["cell_attn"] = torch.cat(cell_level_attn).numpy()

    def plot_losses(self, save: str = None):
        """Plot losses.

        Parameters
        ----------
        save
            If not None, save the plot to this location.
        """
        loss_names = self.module.select_losses_to_plot()
        plt_plot_losses(self.history, loss_names, save)

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/0b802762869c43c9f49e69fe62b1a5a9b5c4dae6/scvi/model/base/_archesmixin.py#L30
    # accessed on 7 November 2022
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: BaseModelClass,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
    ) -> BaseModelClass:

        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )

        attr_dict, _, _ = _get_loaded_data(reference_model, device=device)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("Loaded model is from a different class.")
        if _SETUP_ARGS_KEY not in registry:
            raise ValueError("Saved model does not contain original setup inputs.")

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)
        model.module.load_state_dict(reference_model.module.state_dict())
        model.to_device(device)

        model.module.eval()
        model.is_trained_ = True

        return model
