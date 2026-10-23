import logging

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

log = logging.getLogger(__name__)


class DataModule(LightningDataModule):
    """
    Basic DataModule.
    """

    def __init__(
        self,
        dataset_class: type[Dataset],
        batch_size: int,
        train_val_test_split: list[float | int],
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool = False,
        seed: int = 64,
    ):
        """
        Initialise the datamodule.

        :param dataset_class: The Dataset class we will wrap in this datamodule.
        **This should not be a Dataset object**, the Dataset will be instantiated during `setup()`.
        :type dataset_class: Type[Dataset]
        :param batch_size: How many samples per batch to load.
        :type batch_size: int
        :param train_val_test_split: Lengths or fractions of the training, validation and test splits.
        :type train_val_test_split: List[Union[float, int], Union[float, int], Union[float, int]]
        :param num_workers: How many subprocesses to use for data loading.
        :type num_workers: int
        :param pin_memory: If True, the dataloader will copy Tensors
            into device/CUDA pinned memory before returning them.
        :type pin_memory: bool
        :param persistent_workers: If True, the loader keeps its worker processes alive between
            epochs rather than starting fresh ones, avoiding a stall at every epoch boundary at the
            cost of the workers' memory. Ignored when `num_workers` is 0.
        :type persistent_workers: bool
        :param seed: Seed for data generation.
        :type seed: int
        """

        super().__init__()
        self.save_hyperparameters()

        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # `DataLoader` rejects persistent workers when there are none to persist (`num_workers: 0`)
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        # These will be set in setup
        self.data_train: Dataset | None = None
        self.data_valid: Dataset | None = None
        self.data_test: Dataset | None = None

    def setup(
        self,
        stage: str,
    ):
        """
        Split the dataset. Called by Lightning on every device before fitting, validating or testing.

        The split is seeded, so every device (and every call) gets the same one.

        :param stage: `"fit"`, `"validate"` or `"test"`.
        :type stage: str
        """

        if stage not in ["fit", "validate", "test"]:
            raise ValueError(f"Got an unsupported stage in `setup`: {stage=}")

        self.data_train, self.data_valid, self.data_test = random_split(
            self.dataset_class(),
            self.train_val_test_split,
            generator=torch.Generator().manual_seed(self.seed),
        )
        log.info(
            f"Set up datamodule with {len(self.data_train)} train samples, "
            f"{len(self.data_valid)} validation samples "
            f"and {len(self.data_test)} test samples."
        )

    def _loader(
        self,
        dataset: Dataset | None,
        shuffle: bool,
    ) -> DataLoader:
        if dataset is None:
            raise ValueError("The datamodule has not been set up. Ensure `setup` is called first.")
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_valid, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)
