"""This file prepares config fixtures for other tests."""

import os
import wave
from array import array
from collections.abc import Generator
from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@pytest.fixture()
def atcosim_data_dir(tmp_path: Path) -> Path:
    """Use local ATCOSIM data when available, otherwise create a minimal dataset."""
    local_data_dir_value = os.environ.get("ATCOSIM_DATA_DIR")
    if local_data_dir_value:
        local_data_dir = Path(local_data_dir_value).expanduser()
        required_paths = (
            local_data_dir / "train",
            local_data_dir / "test",
            local_data_dir / "transcriptions" / "train_trans.txt",
            local_data_dir / "transcriptions" / "test_trans.txt",
            local_data_dir / "transcriptions" / "fold0" / "train_trans.txt",
            local_data_dir / "transcriptions" / "fold0" / "val_trans.txt",
        )
        if all(path.exists() for path in required_paths):
            return local_data_dir

    data_dir = tmp_path / "ATCOSIM"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    fold_dir = data_dir / "transcriptions" / "fold0"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    fold_dir.mkdir(parents=True)

    def write_wav(path: Path, length: int) -> None:
        samples = array("h", (index % 100 for index in range(length)))
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(32_000)
            wav_file.writeframes(samples.tobytes())

    train_ids = [f"train_{index}" for index in range(4)]
    val_ids = [f"val_{index}" for index in range(2)]
    test_ids = [f"test_{index}" for index in range(2)]
    for index, utterance_id in enumerate(train_ids + val_ids):
        write_wav(train_dir / f"{utterance_id}.wav", 160 + index)
    for index, utterance_id in enumerate(test_ids):
        write_wav(test_dir / f"{utterance_id}.wav", 180 + index)

    transcriptions = data_dir / "transcriptions"
    (transcriptions / "train_trans.txt").write_text(
        "".join(f"{item} train text\n" for item in train_ids + val_ids), encoding="utf-8"
    )
    (transcriptions / "test_trans.txt").write_text(
        "".join(f"{item} test text\n" for item in test_ids), encoding="utf-8"
    )
    (fold_dir / "train_trans.txt").write_text(
        "".join(f"{item} train text\n" for item in train_ids), encoding="utf-8"
    )
    (fold_dir / "val_trans.txt").write_text(
        "".join(f"{item} validation text\n" for item in val_ids), encoding="utf-8"
    )
    return data_dir


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> Generator[DictConfig, None, None]:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own
    temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.
    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> Generator[DictConfig, None, None]:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own
    temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.
    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
