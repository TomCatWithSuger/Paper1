from pathlib import Path

import pytest
import torch

from src.data.atcosim_datamodule import ATCOSIMDataModule
from src.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [2, 4])
def test_atcosim_datamodule(batch_size: int, atcosim_data_dir: Path) -> None:
    """Test ATCOSIM loading and collation without external dataset dependencies."""
    dm = ATCOSIMDataModule(data_dir=str(atcosim_data_dir), batch_size=batch_size)

    dm.prepare_data()
    dm.setup()

    assert dm.data_train is not None
    assert dm.data_val is not None
    assert dm.data_test is not None
    assert len(dm.data_train) >= batch_size
    assert len(dm.data_val) > 0
    assert len(dm.data_test) > 0

    batch = next(iter(dm.train_dataloader()))
    assert batch["waveforms"].shape[0] == batch_size
    assert batch["waveforms"].dtype == torch.float32
    assert batch["lengths"].shape == (batch_size,)
    assert batch["lengths"].dtype == torch.int64
    assert batch["attention_mask"].shape == batch["waveforms"].shape
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["sample_rate"] == 32_000
    assert len(batch["ids"]) == batch_size
    assert len(batch["texts"]) == batch_size
