from pathlib import Path

import pytest
import torch

from src.data.atcosim_datamodule import ATCOSIMDataModule
from src.data.components.vctk_splits import (
    RatioSpeakerSplit,
    UnseenSpeakerSentenceSplit,
)
from src.data.mnist_datamodule import MNISTDataModule
from src.data.vctk_datamodule import VCTKDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """验证 MNIST 的下载、数据划分、数据加载器、批大小和张量类型。

    :param batch_size: 参数化测试使用的数据加载批大小。
    :return: 无返回值；任一数据数量、形状或类型断言不满足时测试失败。
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
    """验证 ATCOSIM 的数据划分、波形加载和动态批处理结果。

    :param batch_size: 参数化测试使用的数据加载批大小。
    :param atcosim_data_dir: 真实或临时生成的 ATCOSIM 数据根目录。
    :return: 无返回值；数据集、批形状或字段类型不符合预期时测试失败。
    """
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


@pytest.mark.parametrize("batch_size", [2, 4])
def test_vctk_datamodule(batch_size: int, vctk_data_dir: Path) -> None:
    """验证 VCTK 配对、预处理、说话人划分和动态批处理结果。

    :param batch_size: 参数化测试使用的数据加载批大小。
    :param vctk_data_dir: 真实或临时生成的 VCTK 数据根目录。
    :return: 无返回值；数据划分、张量形状或字段内容不符合预期时测试失败。
    """
    dm = VCTKDataModule(
        data_dir=str(vctk_data_dir),
        split_strategy=RatioSpeakerSplit(val_ratio=0.2, test_ratio=0.2),
        batch_size=batch_size,
    )

    dm.prepare_data()
    dm.setup()

    assert dm.data_train is not None
    assert dm.data_val is not None
    assert dm.data_test is not None
    assert len(dm.data_train) >= batch_size
    train_speakers = set(dm.data_train.speaker_ids)
    val_speakers = set(dm.data_val.speaker_ids)
    test_speakers = set(dm.data_test.speaker_ids)
    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)

    batch = next(iter(dm.train_dataloader()))
    assert batch["waveforms"].shape[0] == batch_size
    assert batch["waveforms"].dtype == torch.float32
    assert batch["lengths"].shape == (batch_size,)
    assert batch["lengths"].dtype == torch.int64
    assert batch["attention_mask"].shape == batch["waveforms"].shape
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["sample_rate"] == 22_050
    assert len(batch["ids"]) == batch_size
    assert len(batch["speaker_ids"]) == batch_size
    assert len(batch["texts"]) == batch_size
    assert len(batch["reference_ids"]) == batch_size
    assert all(
        utterance_id != reference_id
        for utterance_id, reference_id in zip(batch["ids"], batch["reference_ids"])
    )
    assert batch["mel_spectrograms"].shape[:2] == (batch_size, 80)
    assert batch["mel_lengths"].shape == (batch_size,)
    assert batch["mel_attention_mask"].shape == (
        batch_size,
        batch["mel_spectrograms"].shape[2],
    )
    assert torch.isfinite(batch["mel_spectrograms"]).all()
    assert dm.data_train.max_samples == 10 * 22_050
    assert "reference_waveforms" not in batch
    assert "reference_mel_spectrograms" not in batch


def test_vctk_unseen_speaker_sentence_split(vctk_data_dir: Path) -> None:
    """验证未见说话人和指定语句划分及跨说话人参考选择。

    :param vctk_data_dir: 真实或临时生成的 VCTK 数据根目录。
    :return: 无返回值；训练、验证、测试集合边界或参考说话人不正确时测试失败。
    """
    evaluation_speakers = {"p229", "p230"}
    evaluation_sentences = {"001", "002"}
    dm = VCTKDataModule(
        data_dir=str(vctk_data_dir),
        split_strategy=UnseenSpeakerSentenceSplit(
            evaluation_speaker_ids=sorted(evaluation_speakers),
            evaluation_sentence_ids=sorted(evaluation_sentences),
            val_ratio=0.5,
        ),
        test_reference_mode="different_speaker",
        batch_size=2,
    )

    dm.prepare_data()
    dm.setup()

    assert dm.data_train is not None
    assert dm.data_val is not None
    assert dm.data_test is not None
    train_speakers = set(dm.data_train.speaker_ids)
    val_speakers = set(dm.data_val.speaker_ids)
    test_speakers = set(dm.data_test.speaker_ids)
    assert train_speakers == val_speakers
    assert train_speakers.isdisjoint(evaluation_speakers)
    assert test_speakers == evaluation_speakers
    assert all(sample.sentence_id not in evaluation_sentences for sample in dm.data_train.samples)
    assert all(sample.sentence_id not in evaluation_sentences for sample in dm.data_val.samples)
    assert all(sample.sentence_id in evaluation_sentences for sample in dm.data_test.samples)

    batch = next(iter(dm.test_dataloader()))
    assert all(
        speaker_id != reference_speaker_id
        for speaker_id, reference_speaker_id in zip(
            batch["speaker_ids"], batch["reference_speaker_ids"]
        )
    )


def test_vctk_cached_meanvc_features(vctk_data_dir: Path, vctk_feature_cache_dir: Path) -> None:
    """验证 VCTK 离线内容特征及目标和参考说话人向量的批处理。

    :param vctk_data_dir: 真实或临时生成的 VCTK 数据根目录。
    :param vctk_feature_cache_dir: 与当前 VCTK 样本匹配的真实或临时 MeanVC 缓存目录。
    :return: 无返回值；缓存张量形状、长度或批处理字段不符合预期时测试失败。
    """
    dm = VCTKDataModule(
        data_dir=str(vctk_data_dir),
        split_strategy=RatioSpeakerSplit(val_ratio=0.2, test_ratio=0.2),
        feature_cache_dir=str(vctk_feature_cache_dir),
        require_feature_cache=True,
        batch_size=2,
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    assert batch["content_features"].shape == (2, 256, 20)
    assert batch["content_feature_lengths"].tolist() == [20, 20]
    assert batch["speaker_features"].shape == (2, 256)
    assert batch["reference_speaker_features"].shape == (2, 256)
