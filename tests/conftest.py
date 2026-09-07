"""模块说明：

- 为测试提供共享数据、实验覆盖参数以及训练/评估配置 fixtures。
- ATCOSIM、VCTK 和 MeanVC 优先使用环境变量指定的真实资源，否则创建临时资源。
- 对外接口由 pytest 自动发现，包括数据 fixtures、``experiment_data_overrides``、
    ``cfg_train`` 和 ``cfg_eval`` 等；本文件不提供独立运行入口。
"""

# ====================
# 1. Imports
# ====================

import os
import wave
from array import array
from collections.abc import Generator
from pathlib import Path

import pytest
import rootutils
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from src.data.atcosim_datamodule import ATCOSIMDataModule
from src.data.components.vctk_splits import discover_vctk_samples
from src.data.vctk_datamodule import VCTKDataModule

# ====================
# 2. Initialization
# pytest 加载本文件时初始化项目根目录
# ====================

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# ====================
# 3. Utilities
# 文件写入与资源有效性检查
# ====================


def _write_wav(path: Path, sample_rate: int, length: int) -> None:
    """写入单声道 16 位测试波形。

    :param path: WAV 文件的输出路径。
    :param sample_rate: 波形采样率，单位为 Hz。
    :param length: 需要生成的采样点数量。
    :return: 无返回值，结果直接写入 ``path``。
    """
    samples = array("h", (index % 100 for index in range(length)))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def _valid_atcosim_dir(data_dir: Path) -> bool:
    """判断目录是否包含测试所需的 ATCOSIM 结构。

    :param data_dir: 待检查的 ATCOSIM 根目录。
    :return: 训练集、测试集及 fold0 转录文件均存在时返回 ``True``。
    """
    required_paths = (
        data_dir / "train",
        data_dir / "test",
        data_dir / "transcriptions" / "train_trans.txt",
        data_dir / "transcriptions" / "test_trans.txt",
        data_dir / "transcriptions" / "fold0" / "train_trans.txt",
        data_dir / "transcriptions" / "fold0" / "val_trans.txt",
    )
    return all(path.exists() for path in required_paths)


def _valid_vctk_dir(data_dir: Path) -> bool:
    """判断目录是否包含测试所需的 VCTK 结构。

    :param data_dir: 待检查的 VCTK 根目录。
    :return: 文本目录和静音裁剪音频目录均存在时返回 ``True``。
    """
    return all(path.is_dir() for path in (data_dir / "txt", data_dir / "wav48_silence_trimmed"))


def _valid_meanvc_cache(cache_dir: Path, data_dir: Path) -> bool:
    """判断 MeanVC 缓存是否覆盖当前 VCTK 的全部样本。

    :param cache_dir: 包含 ``content`` 和 ``speaker`` 子目录的缓存根目录。
    :param data_dir: 用于确定全部语句标识的 VCTK 根目录。
    :return: 每条 VCTK 样本均有内容特征和说话人特征时返回 ``True``。
    """
    if not _valid_vctk_dir(data_dir):
        return False
    samples = discover_vctk_samples(data_dir, "mic1")
    return bool(samples) and all(
        (cache_dir / "content" / f"{sample.utterance_id}.pt").is_file()
        and (cache_dir / "speaker" / f"{sample.utterance_id}.pt").is_file()
        for sample in samples
    )


# ====================
# 4. Internal Functions
# 临时测试资源构造
# ====================


def _create_atcosim_data(data_dir: Path) -> Path:
    """生成可用于加载和快速训练的最小 ATCOSIM 数据集。

    :param data_dir: 临时 ATCOSIM 数据集的目标根目录。
    :return: 已写入音频与转录文件的 ATCOSIM 根目录。
    """
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    fold_dir = data_dir / "transcriptions" / "fold0"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    fold_dir.mkdir(parents=True)

    train_ids = [f"train_{index}" for index in range(4)]
    val_ids = [f"val_{index}" for index in range(2)]
    test_ids = [f"test_{index}" for index in range(2)]
    for index, utterance_id in enumerate(train_ids + val_ids):
        _write_wav(train_dir / f"{utterance_id}.wav", 32_000, 1024 + index)
    for index, utterance_id in enumerate(test_ids):
        _write_wav(test_dir / f"{utterance_id}.wav", 32_000, 1024 + index)

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


def _create_vctk_data(data_dir: Path) -> Path:
    """生成同时满足比例划分和未见说话人语句划分的小型 VCTK 数据集。

    :param data_dir: 临时 VCTK 数据集的目标根目录。
    :return: 已写入音频与文本文件的 VCTK 根目录。
    """
    evaluation_speakers = (
        "p345",
        "p347",
        "p351",
        "p360",
        "p361",
        "p362",
        "p363",
        "p364",
        "p374",
        "p376",
    )
    speakers = tuple(f"p{225 + index}" for index in range(6)) + evaluation_speakers
    sentence_ids = tuple(f"{index:03d}" for index in range(1, 20))

    for speaker_id in speakers:
        transcript_dir = data_dir / "txt" / speaker_id
        audio_dir = data_dir / "wav48_silence_trimmed" / speaker_id
        transcript_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)
        for sentence_id in sentence_ids:
            utterance_id = f"{speaker_id}_{sentence_id}"
            (transcript_dir / f"{utterance_id}.txt").write_text(
                f"Synthetic speech from {speaker_id}.\n", encoding="utf-8"
            )
            _write_wav(audio_dir / f"{utterance_id}_mic1.wav", 48_000, 16_000)
    return data_dir


def _create_meanvc_cache(cache_dir: Path, data_dir: Path) -> Path:
    """为当前 VCTK 样本生成测试用 MeanVC 张量缓存。

    :param cache_dir: 临时 MeanVC 缓存的目标根目录。
    :param data_dir: 提供语句标识的 VCTK 根目录。
    :return: 已生成内容特征和说话人特征的缓存根目录。
    """
    for sample in discover_vctk_samples(data_dir, "mic1"):
        content_path = cache_dir / "content" / f"{sample.utterance_id}.pt"
        speaker_path = cache_dir / "speaker" / f"{sample.utterance_id}.pt"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        speaker_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.randn(256, 20), content_path)
        torch.save(torch.randn(256), speaker_path)
    return cache_dir


# ====================
# 5. Public API
# pytest 对外提供的数据 fixtures
# ====================


@pytest.fixture(scope="package")
def atcosim_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """提供 ATCOSIM 测试数据目录。

    ``ATCOSIM_DATA_DIR`` 指向完整数据集时使用真实数据，否则生成包级共享的临时数据。

    :param tmp_path_factory: pytest 临时目录工厂。
    :return: 可直接传给 ``ATCOSIMDataModule`` 的数据根目录。
    """
    local_data_dir_value = os.environ.get("ATCOSIM_DATA_DIR")
    if local_data_dir_value:
        local_data_dir = Path(local_data_dir_value).expanduser()
        if _valid_atcosim_dir(local_data_dir):
            return local_data_dir
    return _create_atcosim_data(tmp_path_factory.mktemp("atcosim_data") / "ATCOSIM")


@pytest.fixture(scope="package")
def vctk_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """提供 VCTK 测试数据目录。

    ``VCTK_DATA_DIR`` 指向完整数据集时使用真实数据，否则生成包级共享的临时数据。

    :param tmp_path_factory: pytest 临时目录工厂。
    :return: 可直接传给 ``VCTKDataModule`` 的数据根目录。
    """
    local_data_dir_value = os.environ.get("VCTK_DATA_DIR")
    if local_data_dir_value:
        local_data_dir = Path(local_data_dir_value).expanduser()
        if _valid_vctk_dir(local_data_dir):
            return local_data_dir
    return _create_vctk_data(tmp_path_factory.mktemp("vctk_data") / "VCTK")


@pytest.fixture(scope="package")
def vctk_feature_cache_dir(vctk_data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """提供与当前 VCTK 数据匹配的 MeanVC 特征缓存。

    ``MEANVC_FEATURE_CACHE_DIR`` 覆盖全部样本时使用真实缓存，否则生成随机临时缓存。

    :param vctk_data_dir: 当前测试使用的 VCTK 根目录。
    :param tmp_path_factory: pytest 临时目录工厂。
    :return: 包含 ``content`` 和 ``speaker`` 特征的缓存根目录。
    """
    local_cache_dir_value = os.environ.get("MEANVC_FEATURE_CACHE_DIR")
    if local_cache_dir_value:
        local_cache_dir = Path(local_cache_dir_value).expanduser()
        if _valid_meanvc_cache(local_cache_dir, vctk_data_dir):
            return local_cache_dir
    return _create_meanvc_cache(
        tmp_path_factory.mktemp("meanvc_cache") / "meanvc_features", vctk_data_dir
    )


# pytest 对外提供的实验 fixture
@pytest.fixture()
def experiment_data_overrides(experiment_name: str, request: pytest.FixtureRequest) -> list[str]:
    """根据实验的 DataModule 类型生成数据路径覆盖参数。

    ATCOSIM 和 VCTK 实验会取得对应的真实或临时 fixture；其他 DataModule 保持原配置。
    注入路径后再次实例化并调用 ``prepare_data()``，未适配且缺少资源的实验会被跳过。

    :param experiment_name: ``configs/experiment`` 下的实验配置名称。
    :param request: 用于按 DataModule 类型动态获取 pytest fixture 的请求对象。
    :return: 传给 Hydra 训练命令的数据路径覆盖参数列表。
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={experiment_name}"])
        datamodule = instantiate(cfg.data)

    data_overrides: list[str] = []
    if isinstance(datamodule, ATCOSIMDataModule):
        data_dir = request.getfixturevalue("atcosim_data_dir")
        data_overrides.append(f"data.data_dir={data_dir}")
    elif isinstance(datamodule, VCTKDataModule):
        data_dir = request.getfixturevalue("vctk_data_dir")
        cache_dir = request.getfixturevalue("vctk_feature_cache_dir")
        data_overrides.extend((f"data.data_dir={data_dir}", f"data.feature_cache_dir={cache_dir}"))

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"experiment={experiment_name}", *data_overrides],
        )
        datamodule = instantiate(cfg.data)

    try:
        datamodule.prepare_data()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return data_overrides


# pytest 对外提供的配置 fixtures
@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """创建包级共享的默认训练配置。

    :return: 已限制训练轮数、批次数和设备数量的 Hydra 训练配置。
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # 统一限制测试开销并关闭非必要输出
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
    """创建包级共享的默认评估配置。

    :return: 已限制测试批次和设备数量的 Hydra 评估配置。
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # 统一限制测试开销并关闭非必要输出
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
    """为单个测试复制训练配置并设置独立输出目录。

    :param cfg_train_global: 包级共享的默认训练配置。
    :param tmp_path: 当前测试独占的临时目录。
    :return: 输出目录和日志目录均指向 ``tmp_path`` 的训练配置生成器。
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> Generator[DictConfig, None, None]:
    """为单个测试复制评估配置并设置独立输出目录。

    :param cfg_eval_global: 包级共享的默认评估配置。
    :param tmp_path: 当前测试独占的临时目录。
    :return: 输出目录和日志目录均指向 ``tmp_path`` 的评估配置生成器。
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
