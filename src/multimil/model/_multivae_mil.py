import logging
import warnings
from typing import Union

import anndata as ad
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
from multimil.model import MILClassifier, MultiVAE
from multimil.module import MultiVAETorch_MIL
from multimil.utils import (
    calculate_size_factor,
    get_bag_info,
    get_predictions,
    plt_plot_losses,
    prep_minibatch,
    save_predictions_in_adata,
    select_covariates,
    setup_ordinal_regression,
)

logger = logging.getLogger(__name__)


class MultiVAE_MIL(BaseModelClass, ArchesMixin):
    """MultiVAE_MIL model.

    Parameters
    ----------
    adata
        Annotated data object.
    sample_key
        Key for the sample column in the adata object.
    classification
        List of keys for the categorical covariates used for classification.
    regression
        List of keys for the continuous covariates used for regression.
    ordinal_regression
        List of keys for the ordinal covariates used for ordinal regression.
    sample_batch_size
        Bag batch size for training the model.
    integrate_on
        Key for the covariate used for integration.
    condition_encoders
        Whether to condition the encoders on the covariates.
    condition_decoders
        Whether to condition the decoders on the covariates.
    normalization
        Type of normalization to be applied.
    n_layers_encoders
        Number of layers in the encoders.
    n_layers_decoders
        Number of layers in the decoders.
    n_hidden_encoders
        Number of hidden units in the encoders.
    n_hidden_decoders
        Number of hidden units in the decoders.
    z_dim
        Dimensionality of the latent space.
    losses
        List of loss functions to be used.
    dropout
        Dropout rate.
    cond_dim
        Dimensionality of the covariate embeddings.
    kernel_type
        Type of kernel to be used in MMD calculation.
    loss_coefs
        List of coefficients for different losses.
    scoring
        Scoring method for the MIL classifier.
    attn_dim
        Dimensionality of the hidden dimentino in the attention mechanism.
    n_layers_cell_aggregator
        Number of layers in the cell aggregator.
    n_layers_classifier
        Number of layers in the classifier.
    n_layers_regressor
        Number of layers in the regressor.
    n_layers_mlp_att
        Number of layers in the MLP attention mechanism.
    n_layers_cont_embed
        Number of layers in the continuous embedding.
    n_hidden_cell_aggregator
        Number of hidden units in the cell aggregator.
    n_hidden_classifier
        Number of hidden units in the classifier.
    n_hidden_cont_embed
        Number of hidden units in the continuous embedding.
    n_hidden_mlp_attn
        Number of hidden units in the MLP attention mechanism.
    n_hidden_regressor
        Number of hidden units in the regressor.
    class_loss_coef
        Coefficient for the classification loss.
    regression_loss_coef
        Coefficient for the regression loss.
    cont_cov_type
        How to calucate the embeddings for the continuous covariates.
    mmd
        Type of maximum mean discrepancy.
    sample_in_vae
        Whether to include the sample key in the VAE as a covariate.
    activation
        Activation function to be used.
    initialization
        Initialization method for the weights.
    anneal_class_loss
        Whether to anneal the classification loss.
    """

    def __init__(
        self,
        adata,
        sample_key,
        classification=None,
        regression=None,
        ordinal_regression=None,
        sample_batch_size=128,
        integrate_on=None,
        condition_encoders=False,
        condition_decoders=True,
        normalization="layer",
        n_layers_encoders=None,
        n_layers_decoders=None,
        n_hidden_encoders=None,
        n_hidden_decoders=None,
        z_dim=30,
        losses=None,
        dropout=0.2,
        cond_dim=16,
        kernel_type="gaussian",
        loss_coefs=None,
        scoring="gated_attn",
        attn_dim=16,
        n_layers_cell_aggregator: int = 1,
        n_layers_classifier: int = 2,
        n_layers_regressor: int = 1,
        n_layers_mlp_attn: int = 1,
        n_layers_cont_embed: int = 1,
        n_hidden_cell_aggregator: int = 128,
        n_hidden_classifier: int = 128,
        n_hidden_cont_embed: int = 128,
        n_hidden_mlp_attn: int = 32,
        n_hidden_regressor: int = 128,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        cont_cov_type="logsigm",
        mmd="latent",
        sample_in_vae=True,
        activation="leaky_relu",  # or tanh
        initialization="kaiming",  # xavier (tanh) or kaiming (leaky_relu)
        anneal_class_loss=False,
    ):
        super().__init__(adata)

        if classification is None:
            classification = []
        if regression is None:
            regression = []
        if ordinal_regression is None:
            ordinal_regression = []

        # add prediction covariates to ignore_covariates_vae
        ignore_covariates_vae = []
        if sample_in_vae is False:
            ignore_covariates_vae.append(sample_key)
        for key in classification + ordinal_regression + regression:
            ignore_covariates_vae.append(key)

        setup_args = self.adata_manager.registry["setup_args"].copy()

        setup_args.pop("ordinal_regression_order")
        MultiVAE.setup_anndata(
            adata,
            **setup_args,
        )

        self.multivae = MultiVAE(
            adata=adata,
            integrate_on=integrate_on,
            condition_encoders=condition_encoders,
            condition_decoders=condition_decoders,
            normalization=normalization,
            z_dim=z_dim,
            losses=losses,
            dropout=dropout,
            cond_dim=cond_dim,
            kernel_type=kernel_type,
            loss_coefs=loss_coefs,
            cont_cov_type=cont_cov_type,
            n_layers_cont_embed=n_layers_cont_embed,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_hidden_cont_embed=n_hidden_cont_embed,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            mmd=mmd,
            activation=activation,
            initialization=initialization,
            ignore_covariates=ignore_covariates_vae,
        )

        # add all actual categorical covariates to ignore_covariates_mil
        ignore_covariates_mil = []
        if self.adata_manager.registry["setup_args"]["categorical_covariate_keys"] is not None:
            for cat_name in self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
                if cat_name not in classification + ordinal_regression + [sample_key]:
                    ignore_covariates_mil.append(cat_name)
        # add all actual continuous covariates to ignore_covariates_mil
        if self.adata_manager.registry["setup_args"]["continuous_covariate_keys"] is not None:
            for cat_name in self.adata_manager.registry["setup_args"]["continuous_covariate_keys"]:
                if cat_name not in regression:
                    ignore_covariates_mil.append(cat_name)

        setup_args = self.adata_manager.registry["setup_args"].copy()
        # setup_args.pop('batch_key')
        setup_args.pop("size_factor_key")
        setup_args.pop("rna_indices_end")

        MILClassifier.setup_anndata(
            adata=adata,
            **setup_args,
        )

        self.mil = MILClassifier(
            adata=adata,
            sample_key=sample_key,
            classification=classification,
            regression=regression,
            ordinal_regression=ordinal_regression,
            sample_batch_size=sample_batch_size,
            normalization=normalization,
            z_dim=z_dim,
            dropout=dropout,
            scoring=scoring,
            attn_dim=attn_dim,
            n_layers_cell_aggregator=n_layers_cell_aggregator,
            n_layers_classifier=n_layers_classifier,
            n_layers_mlp_attn=n_layers_mlp_attn,
            n_layers_regressor=n_layers_regressor,
            n_hidden_regressor=n_hidden_regressor,
            n_hidden_cell_aggregator=n_hidden_cell_aggregator,
            n_hidden_classifier=n_hidden_classifier,
            n_hidden_mlp_attn=n_hidden_mlp_attn,
            class_loss_coef=class_loss_coef,
            regression_loss_coef=regression_loss_coef,
            activation=activation,
            initialization=initialization,
            anneal_class_loss=anneal_class_loss,
            ignore_covariates=ignore_covariates_mil,
        )
        # TODO check if don't have to set these to None, rather reference them in self.module
        # i.e. create a MultiVAETorch_MIL, it will create some module inside, but then do
        # self.module.vae_module = self.multivae.module and
        # self.module.mil_module = self.mil.module

        # clear up memory
        self.multivae.module = None
        self.mil.module = None

        self.sample_in_vae = sample_in_vae

        self.module = MultiVAETorch_MIL(
            # vae
            modality_lengths=self.multivae.modality_lengths,
            condition_encoders=condition_encoders,
            condition_decoders=condition_decoders,
            normalization=normalization,
            activation=activation,
            initialization=initialization,
            z_dim=z_dim,
            losses=losses,
            dropout=dropout,
            cond_dim=cond_dim,
            kernel_type=kernel_type,
            loss_coefs=loss_coefs,
            num_groups=self.multivae.num_groups,
            integrate_on_idx=self.multivae.integrate_on_idx,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_layers_cont_embed=n_layers_cont_embed,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            n_hidden_cont_embed=n_hidden_cont_embed,
            cat_covariate_dims=self.multivae.cat_covariate_dims,
            cont_covariate_dims=self.multivae.cont_covariate_dims,
            cat_covs_idx=self.multivae.cat_covs_idx,
            cont_covs_idx=self.multivae.cont_covs_idx,
            cont_cov_type=cont_cov_type,
            mmd=mmd,
            # mil
            num_classification_classes=self.mil.num_classification_classes,
            scoring=scoring,
            attn_dim=attn_dim,
            n_layers_cell_aggregator=n_layers_cell_aggregator,
            n_layers_classifier=n_layers_classifier,
            n_layers_mlp_attn=n_layers_mlp_attn,
            n_layers_regressor=n_layers_regressor,
            n_hidden_regressor=n_hidden_regressor,
            n_hidden_cell_aggregator=n_hidden_cell_aggregator,
            n_hidden_classifier=n_hidden_classifier,
            n_hidden_mlp_attn=n_hidden_mlp_attn,
            class_loss_coef=class_loss_coef,
            regression_loss_coef=regression_loss_coef,
            sample_batch_size=sample_batch_size,
            class_idx=self.mil.class_idx,
            ord_idx=self.mil.ord_idx,
            reg_idx=self.mil.regression_idx,
            anneal_class_loss=anneal_class_loss,
        )

        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int = 200,
        lr: float = 5e-4,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_epochs_kl_warmup: int | None = None,
        n_steps_kl_warmup: int | None = None,
        adversarial_mixing: bool = False,
        plan_kwargs: dict | None = None,
        early_stopping_monitor: str | None = "accuracy_validation",
        early_stopping_mode: str | None = "max",
        save_checkpoint_every_n_epochs: int | None = None,
        path_to_checkpoints: str | None = None,
        **kwargs,
    ):
        """Trains the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        weight_decay
            weight decay regularization term for optimization
        eps
            Optimizer eps
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        save_best
            Save the best model state with respect to the validation loss, or use the final
            state in the training procedure
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`.
            If so, val is checked every epoch.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`. Default is 1/3 of `max_epochs`.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        adversarial_mixing
            Whether to use adversarial mixing in the training procedure.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        early_stopping_monitor
            Metric to monitor for early stopping. Default is "accuracy_validation".
        early_stopping_mode
            One of "min" or "max". Default is "max".
        save_checkpoint_every_n_epochs
            Save a checkpoint every n epochs.
        path_to_checkpoints
            Path to save checkpoints.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.

        Returns
        -------
        Trainer object.
        """
        if len(self.mil.regression) > 0:
            if early_stopping_monitor == "accuracy_validation":
                warnings.warn(
                    "Setting early_stopping_monitor to 'regression_loss_validation' and early_stopping_mode to 'min' as regression is used.",
                    stacklevel=2,
                )
                early_stopping_monitor = "regression_loss_validation"
                early_stopping_mode = "min"
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
            if "callbacks" not in kwargs.keys():
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
                    f"`save_checkpoint_every_n_epochs` = {save_checkpoint_every_n_epochs} so `path_to_checkpoints` has to be not None but is {path_to_checkpoints}."
                )

        data_splitter = GroupDataSplitter(
            self.adata_manager,
            group_column=self.mil.sample_key,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            accelerator=accelerator,
            device=device,
        )
        training_plan = AdversarialTrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            device=device,
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
        adata: ad.AnnData,
        size_factor_key: str | None = None,
        rna_indices_end: int | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        ordinal_regression_order: dict[str, list[str]] | None = None,
        **kwargs,
    ):
        """Set up :class:`~anndata.AnnData` object.

        A mapping will be created between data fields used by ``scvi`` to their respective locations in adata.
        This method will also compute the log mean and log variance per batch for the library size prior.
        None of the data in adata are modified. Only adds fields to adata.

        Parameters
        ----------
        adata
            AnnData object containing raw counts. Rows represent cells, columns represent features.
        size_factor_key
            Key in `adata.obs` containing the size factor. If `None`, will be calculated from the RNA counts.
        rna_indices_end
            Integer to indicate where RNA feature end in the AnnData object. May be needed to calculate ``libary_size``.
        categorical_covariate_keys
            Keys in `adata.obs` that correspond to categorical data.
        continuous_covariate_keys
            Keys in `adata.obs` that correspond to continuous data.
        ordinal_regression_order
            Dictionary with regression classes as keys and order of classes as values.
        kwargs
            Additional parameters to pass to register_fields() of AnnDataManager.
        """
        setup_ordinal_regression(adata, ordinal_regression_order, categorical_covariate_keys)

        setup_method_args = cls._get_setup_method_args(**locals())

        # TODO first add from multivae, then from mil
        anndata_fields = [
            fields.LayerField(
                REGISTRY_KEYS.X_KEY,
                layer=None,
            ),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, None),
            fields.CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        size_factor_key = calculate_size_factor(adata, size_factor_key, rna_indices_end)
        anndata_fields.append(fields.NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key))

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_model_output(
        self,
        adata=None,
        batch_size=256,
    ):
        """Save the latent representation, attention scores and predictions in the adata object.

        Parameters
        ----------
        adata
            AnnData object to run the model on. If `None`, the model's AnnData object is used.
        batch_size
            Minibatch size to use. Default is 256.

        """
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata,
            batch_size=batch_size,
            min_size_per_class=batch_size,  # hack to ensure that not full batches are processed properly
            data_loader_class=GroupAnnDataLoader,
            shuffle=False,
            shuffle_classes=False,
            group_column=self.mil.sample_key,
            drop_last=False,
        )

        latent, cell_level_attn, bags = [], [], []
        class_pred, ord_pred, reg_pred = {}, {}, {}
        (
            bag_class_true,
            bag_class_pred,
            bag_reg_true,
            bag_reg_pred,
            bag_ord_true,
            bag_ord_pred,
        ) = ({}, {}, {}, {}, {}, {})

        bag_counter = 0
        cell_counter = 0

        for tensors in scdl:
            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            pred = outputs["predictions"]

            latent += [z.cpu()]

            cell_attn = self.module.mil_module.cell_level_aggregator[-1].A.squeeze(dim=1)
            sample_size = cell_attn.shape[-1]
            cell_attn = cell_attn.flatten()  # in inference always one patient per batch
            cell_level_attn += [cell_attn.cpu()]

            minibatch_size, n_samples_in_batch = prep_minibatch(cat_covs, self.module.mil_module.sample_batch_size)
            regression = select_covariates(cont_covs, self.mil.regression_idx, n_samples_in_batch)
            ordinal_regression = select_covariates(cat_covs, self.mil.ord_idx, n_samples_in_batch)
            classification = select_covariates(cat_covs, self.mil.class_idx, n_samples_in_batch)

            bag_class_pred, bag_class_true, class_pred = get_predictions(
                self.mil.class_idx, pred, classification, sample_size, bag_class_pred, bag_class_true, class_pred
            )
            bag_ord_pred, bag_ord_true, ord_pred = get_predictions(
                self.mil.ord_idx,
                pred,
                ordinal_regression,
                sample_size,
                bag_ord_pred,
                bag_ord_true,
                ord_pred,
                len(self.mil.class_idx),
            )
            bag_reg_pred, bag_reg_true, reg_pred = get_predictions(
                self.mil.regression_idx,
                pred,
                regression,
                sample_size,
                bag_reg_pred,
                bag_reg_true,
                reg_pred,
                len(self.mil.class_idx) + len(self.mil.ord_idx),
            )

            bags, cell_counter, bag_counter = get_bag_info(
                bags,
                n_samples_in_batch,
                minibatch_size,
                cell_counter,
                bag_counter,
                self.module.mil_module.sample_batch_size,
            )

        latent = torch.cat(latent).numpy()
        cell_level = torch.cat(cell_level_attn).numpy()

        adata.obsm["X_multiMIL"] = latent
        adata.obs["cell_attn"] = cell_level
        flat_bags = [value for sublist in bags for value in sublist]
        adata.obs["bags"] = flat_bags

        for i in range(len(self.mil.class_idx)):
            name = self.mil.classification[i]
            class_names = self.adata_manager.get_state_registry("extra_categorical_covs")["mappings"][name]
            save_predictions_in_adata(
                adata,
                i,
                self.mil.classification,
                bag_class_pred,
                bag_class_true,
                class_pred,
                class_names,
                name,
                clip="argmax",
            )
        for i in range(len(self.mil.ord_idx)):
            name = self.mil.ordinal_regression[i]
            class_names = self.adata_manager.get_state_registry("extra_categorical_covs")["mappings"][name]
            save_predictions_in_adata(
                adata,
                i,
                self.mil.ordinal_regression,
                bag_ord_pred,
                bag_ord_true,
                ord_pred,
                class_names,
                name,
                clip="clip",
            )
        for i in range(len(self.mil.regression_idx)):
            name = self.mil.regression[i]
            reg_names = self.adata_manager.get_state_registry("extra_continuous_covs")["columns"]
            save_predictions_in_adata(
                adata,
                i,
                self.mil.regression,
                bag_reg_pred,
                bag_reg_true,
                reg_pred,
                reg_names,
                name,
                clip=None,
                reg=True,
            )

    def plot_losses(self, save=None):
        """Plot losses.

        Parameters
        ----------
        save
            If not None, save the plot to this location.
        """
        loss_names = []
        loss_names.extend(self.module.vae_module.select_losses_to_plot())
        loss_names.extend(self.module.mil_module.select_losses_to_plot())
        plt_plot_losses(self.history, loss_names, save)

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/0b802762869c43c9f49e69fe62b1a5a9b5c4dae6/scvi/model/base/_archesmixin.py#L30
    # accessed on 7 November 2022
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData, #ad.AnnData ???
        reference_model: BaseModelClass,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        freeze: bool = True,
        ignore_covariates: list[str] | None = None,
    ) -> BaseModelClass:
        """Online update of a reference model with scArches algorithm # TODO cite.

        Parameters
        ----------
        adata
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the ``registry``.
        reference_model
            Already instantiated model of the same class.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).
        freeze
            Whether to freeze the encoders and decoders and only train the new weights.
        ignore_covariates
            List of covariates to ignore. Needed for query-to-reference mapping. Default is `None`.

        Returns
        -------
        Model with updated architecture and weights.
        """

        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )

        attr_dict, _, _ = _get_loaded_data(reference_model, device=device)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError("Saved model does not contain original setup inputs. " "Cannot load the original setup.")

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)

        # set up vae model with load_query_data
        ref_adata = reference_model.adata.copy()
        setup_args = reference_model.adata_manager.registry["setup_args"].copy()
        setup_args.pop("ordinal_regression_order")

        MultiVAE.setup_anndata(ref_adata, **setup_args)

        ignore_covariates = []
        if reference_model.sample_in_vae is False:
            ignore_covariates.append(reference_model.mil.sample_key)

        # needed for the load_query_data to work
        reference_model.multivae.module = reference_model.module.vae_module.to(device)
        new_vae = MultiVAE.load_query_data(
            adata,
            reference_model=reference_model.multivae,
            accelerator=accelerator,
            device=device,
            freeze=freeze,
            ignore_covariates=ignore_covariates
            + reference_model.mil.classification
            + reference_model.mil.regression
            + reference_model.mil.ordinal_regression,
        )

        # clear up memory
        reference_model.multivae.module = None

        # set modules in the new model
        model.module.vae_module = new_vae.module
        model.module.mil_module = reference_model.module.mil_module

        if freeze is True:
            for name, p in model.module.named_parameters():
                if "mil" in name:
                    p.requires_grad = False

        model.module.eval()
        model.is_trained_ = False

        model.to_device(device)

        return model

    def train_vae(
        self,
        max_epochs: int = 200,
        lr: float = 1e-4,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 256,
        weight_decay: float = 0,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_epochs_kl_warmup: int | None = None,
        n_steps_kl_warmup: int | None = None,
        adversarial_mixing: bool = False,
        plan_kwargs: dict | None = None,
        plot_losses=True,
        save_loss=None,
        save_checkpoint_every_n_epochs: int | None = None,
        path_to_checkpoints: str | None = None,
        **kwargs,
    ):
        """Train the VAE part of the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        weight_decay
            weight decay regularization term for optimization
        eps
            Optimizer eps.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        save_best
            Save the best model state with respect to the validation loss, or use the final
            state in the training procedure.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`.
            If so, val is checked every epoch.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`. Default is 1/3 of `max_epochs`.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        adversarial_mixing
            Whether to use adversarial mixing in the training procedure.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        plot_losses
            Whether to plot the losses.
        save_loss
            If not None, save the plot to this location.
        save_checkpoint_every_n_epochs
            Save a checkpoint every n epochs.
        path_to_checkpoints
            Path to save checkpoints.
        kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        # TODO add a check if there are any new params added in load_query_data, i.e. if there are any new params that can be trained
        vae = self.multivae
        vae.module = self.module.vae_module

        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )

        vae.train(
            max_epochs=max_epochs,
            lr=lr,
            accelerator=accelerator,
            device=device,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            weight_decay=weight_decay,
            eps=eps,
            early_stopping=early_stopping,
            save_best=save_best,
            check_val_every_n_epoch=check_val_every_n_epoch,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            n_steps_kl_warmup=n_steps_kl_warmup,
            adversarial_mixing=adversarial_mixing,
            plan_kwargs=plan_kwargs,
            save_checkpoint_every_n_epochs=save_checkpoint_every_n_epochs,
            path_to_checkpoints=path_to_checkpoints,
            **kwargs,
        )

        if plot_losses is True:
            vae.plot_losses(save=save_loss)

        self.module.vae_module = vae.module
        self.multivae.module = None
        self.is_trained_ = True

        # otherwise mil module stays on cpu, but vae module is on gpu -> error in inference
        self.to_device(device)
