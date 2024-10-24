from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_device_args
from typing import Optional, Union

class GroupDataSplitter(DataSplitter):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_size < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata_manager : AnnDataManager
        AnnDataManager object that has been created via ``setup_anndata``.
    group_column : str
        Column in AnnData.obs that contains group labels.
    train_size : float, optional
        Proportion of cells to use as the train set, by default 0.9.
    validation_size : Optional[float], optional
        Proportion of cells to use as the validation set, by default None. If None, is set to 1 - ``train_size``.
    **kwargs
        Keyword arguments for data loader. Data loader class is :class:`~mtg.dataloaders.GroupAnnDataLoader`.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        group_column: str,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        **kwargs,
    ):
        self.group_column = group_column
        super().__init__(adata_manager, train_size, validation_size, **kwargs)

    def _create_dataloader(self, indices, shuffle: bool):
        """Helper function to create GroupAnnDataLoader."""
        if len(indices) > 0:
            return GroupAnnDataLoader(
                self.adata_manager,
                self.group_column,
                indices=indices,
                shuffle=shuffle,
                drop_last=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        return None

    def train_dataloader(self):
        """Return data loader for train AnnData."""
        return self._create_dataloader(self.train_idx, shuffle=True)

    def val_dataloader(self):
        """Return data loader for validation AnnData."""
        return self._create_dataloader(self.val_idx, shuffle=False)

    def test_dataloader(self):
        """Return data loader for test AnnData."""
        return self._create_dataloader(self.test_idx, shuffle=False)
