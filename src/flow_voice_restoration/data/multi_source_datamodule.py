import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset, random_split

log = logging.getLogger(__name__)


def resolve_fractions(fractions: Sequence[float | None]) -> list[float]:
    """
    Fill in the one source left as `None` with whatever share the others leave.

    :param fractions: Each source's share of the samples drawn, at most one of them `None`.
    :type fractions: Sequence[float | None]

    :returns: Every source's share, summing to 1.
    :rtype: list[float]
    """

    unset = [index for index, fraction in enumerate(fractions) if fraction is None]
    if len(unset) > 1:
        raise ValueError(f"At most one source may leave its fraction unset, got {len(unset)}.")
    given = [fraction for fraction in fractions if fraction is not None]
    if any(fraction < 0 for fraction in given):
        raise ValueError(f"Fractions must be non-negative, got {list(fractions)}.")
    total = sum(given)
    if unset:
        if total >= 1.0:
            raise ValueError(f"The set fractions sum to {total:g}, leaving nothing for the unset source.")
        return [1.0 - total if fraction is None else fraction for fraction in fractions]
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"Fractions must sum to 1 when none is left unset, got {total:g}.")
    return given


class MultiSourceSampler(Sampler[int]):
    """
    Draw indices into a `ConcatDataset` of several sources, each at a fixed share of every epoch.

    The order depends only on `seed` and the epoch, so every rank derives the same one and takes its
    own stride of it. This replaces Lightning's distributed sampler, so train with
    `trainer.use_distributed_sampler=false`.
    """

    def __init__(
        self,
        source_sizes: Sequence[int],
        fractions: Sequence[float],
        num_samples: int,
        seed: int,
        num_replicas: int,
        rank: int,
    ):
        """
        Initialise the sampler.

        :param source_sizes: Number of items in each source, in `ConcatDataset` order.
        :type source_sizes: Sequence[int]
        :param fractions: Each source's share of an epoch, summing to 1 (see `resolve_fractions`).
        :type fractions: Sequence[float]
        :param num_samples: Draws per epoch across all ranks.
        :type num_samples: int
        :param seed: Seed of the per-epoch order.
        :type seed: int
        :param num_replicas: Ranks the epoch is split across.
        :type num_replicas: int
        :param rank: This process's rank.
        :type rank: int
        """

        if len(source_sizes) != len(fractions):
            raise ValueError(f"Got {len(source_sizes)} source sizes but {len(fractions)} fractions.")
        if any(fraction > 0 and size == 0 for size, fraction in zip(source_sizes, fractions, strict=True)):
            raise ValueError("A source with a non-zero fraction has no items.")

        self.source_sizes = list(source_sizes)
        self.fractions = list(fractions)
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples_per_replica = math.ceil(num_samples / num_replicas)
        self.total_size = self.num_samples_per_replica * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def source_counts(self) -> list[int]:
        """
        How many of an epoch's draws each source gets: its share, rounded by largest remainder.
        """

        exact = [fraction * self.total_size for fraction in self.fractions]
        counts = [math.floor(value) for value in exact]
        by_remainder = sorted(range(len(exact)), key=lambda index: exact[index] - counts[index], reverse=True)
        for index in by_remainder[: self.total_size - sum(counts)]:
            counts[index] += 1
        return counts

    def epoch_order(self) -> torch.Tensor:
        """
        The whole epoch's indices into the `ConcatDataset`, across every rank.

        Each source serves its draws from fresh permutations of its items, so it repeats an item only
        after all of them have come up. Draws from the sources are then shuffled together.
        """

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws, offset = [], 0
        for size, count in zip(self.source_sizes, self.source_counts(), strict=True):
            if count:
                permutations = [torch.randperm(size, generator=generator) for _ in range(math.ceil(count / size))]
                draws.append(torch.cat(permutations)[:count] + offset)
            offset += size
        order = torch.cat(draws)
        return order[torch.randperm(len(order), generator=generator)]

    def __iter__(self) -> Iterator[int]:
        return iter(self.epoch_order()[self.rank :: self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples_per_replica


class MultiSourceDataModule(LightningDataModule):
    """
    Train on several datasets at once, each at a configured share of every epoch, and validate on each
    separately.

    Each source is split exactly as `DataModule` splits its one dataset, so a source that is also some
    `DataModule` run's whole dataset keeps that run's validation set. Validation and testing return one
    loader per source, in source order; the first source is the reference.
    """

    def __init__(
        self,
        sources: Mapping[str, Mapping],
        batch_size: int,
        train_val_test_split: list[float | int],
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool = False,
        seed: int = 64,
        samples_per_epoch: int | None = None,
    ):
        """
        Initialise the datamodule.

        :param sources: Source name to `{"dataset": <dataset factory>, "fraction": <share or None>}`, in order;
            the first is the reference. At most one fraction may be `None`, taking whatever the others leave.
        :type sources: Mapping[str, Mapping]
        :param batch_size: How many samples per batch to load.
        :type batch_size: int
        :param train_val_test_split: Lengths or fractions of each source's training, validation and test splits.
        :type train_val_test_split: list[float | int]
        :param num_workers: How many subprocesses each loader uses.
        :type num_workers: int
        :param pin_memory: Whether loaders copy batches into pinned memory.
        :type pin_memory: bool
        :param persistent_workers: Whether loaders keep their workers between epochs. Ignored when
            `num_workers` is 0.
        :type persistent_workers: bool
        :param seed: Seed of every source's split and of the training order.
        :type seed: int
        :param samples_per_epoch: Training draws per epoch, across all ranks. Defaults to the size of
            the reference source's training split, so an epoch, and every schedule counted in them,
            means what it did when that source was trained on alone.
        :type samples_per_epoch: int | None
        """

        super().__init__()
        self.save_hyperparameters()

        if not sources:
            raise ValueError("Give at least one source.")
        self.source_names = list(sources)
        self.dataset_factories: list[Callable[[], Dataset]] = [source["dataset"] for source in sources.values()]
        self.fractions = resolve_fractions([source.get("fraction") for source in sources.values()])
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # `DataLoader` rejects persistent workers when there are none to persist (`num_workers: 0`)
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.samples_per_epoch = samples_per_epoch
        # These will be set in setup, one entry per source
        self.data_train: list[Subset] | None = None
        self.data_valid: list[Subset] | None = None
        self.data_test: list[Subset] | None = None

    def setup(
        self,
        stage: str,
    ):
        """
        Split every source. Called by Lightning on every device before fitting, validating or testing.

        :param stage: `"fit"`, `"validate"` or `"test"`.
        :type stage: str
        """

        if stage not in ["fit", "validate", "test"]:
            raise ValueError(f"Got an unsupported stage in `setup`: {stage=}")

        splits = [
            random_split(factory(), self.train_val_test_split, generator=torch.Generator().manual_seed(self.seed))
            for factory in self.dataset_factories
        ]
        self.data_train, self.data_valid, self.data_test = (list(split) for split in zip(*splits, strict=True))
        for name, fraction, train, valid, test in zip(
            self.source_names, self.fractions, self.data_train, self.data_valid, self.data_test, strict=True
        ):
            log.info(
                f"Source {name!r} at {fraction:.1%} of training draws: {len(train)} train, "
                f"{len(valid)} validation and {len(test)} test samples."
            )

    @staticmethod
    def _require_setup(datasets: list[Subset] | None) -> list[Subset]:
        if datasets is None:
            raise ValueError("The datamodule has not been set up. Ensure `setup` is called first.")
        return datasets

    def _loader(
        self,
        dataset: Dataset,
        sampler: Sampler | None = None,
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def train_dataloader(self) -> DataLoader:
        data_train = self._require_setup(self.data_train)

        num_replicas, rank = 1, 0
        if self.trainer is not None:
            num_replicas, rank = self.trainer.world_size, self.trainer.global_rank
            # Not public, but nothing else says whether Lightning will wrap this sampler again
            if num_replicas > 1 and getattr(self.trainer._accelerator_connector, "use_distributed_sampler", False):
                raise ValueError(
                    "`MultiSourceDataModule` shards its own training draws across ranks; set "
                    "`trainer.use_distributed_sampler=false` so Lightning does not shard them again."
                )
        sampler = MultiSourceSampler(
            [len(dataset) for dataset in data_train],
            self.fractions,
            num_samples=self.samples_per_epoch or len(data_train[0]),
            seed=self.seed,
            num_replicas=num_replicas,
            rank=rank,
        )
        return self._loader(ConcatDataset(data_train), sampler)

    def val_dataloader(self) -> list[DataLoader]:
        return [self._loader(dataset) for dataset in self._require_setup(self.data_valid)]

    def test_dataloader(self) -> list[DataLoader]:
        return [self._loader(dataset) for dataset in self._require_setup(self.data_test)]
