"""语音流方法共用的批适配、评估和检查点入口。"""

from collections import OrderedDict
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MeanMetric

from src.optim.warmup_cosine import WarmupCosineLR


class VoiceFlowModule(LightningModule):
    """管理条件语音流的批适配、固定采样评估和优化流程。"""

    _TEST_SEED_OFFSET = 1_000_000_000

    def __init__(
        self,
        method,
        optimizer,
        scheduler=None,
        compile=False,
        validation_sampling_count=None,
        test_sampling_count=None,
        evaluation_seed=12345,
        warmup_steps=None,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["method", "optimizer", "scheduler"],
        )
        self.method = method
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.optimization_loss = MeanMetric()
        self.raw_meanflow_mse = MeanMetric()
        self.velocity_mse = MeanMetric()
        self.nonzero_interval_mse = MeanMetric()
        self.jvp_correction_rms = MeanMetric()
        self.one_step_reconstruction_l1 = MeanMetric()
        for count in (validation_sampling_count, test_sampling_count):
            if count is not None and count <= 0:
                raise ValueError("验证和测试采样次数必须为正数")
        if evaluation_seed < 0:
            raise ValueError("evaluation_seed 不能为负数")
        if warmup_steps is not None and warmup_steps < 0:
            raise ValueError("warmup_steps 不能为负数")
        self.validation_sampling_count = validation_sampling_count
        self.test_sampling_count = test_sampling_count
        self.evaluation_seed = evaluation_seed
        self.warmup_steps = warmup_steps

    @staticmethod
    def _tensor(batch, primary_key, fallback_key=None):
        value = batch.get(primary_key)
        if value is None and fallback_key is not None:
            value = batch.get(fallback_key)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"batch tensor not found: {primary_key}")
        return value

    def _training_data(self, batch):
        return (
            self._tensor(batch, "target_mel_spectrograms", "mel_spectrograms"),
            self._tensor(batch, "target_mel_attention_mask", "mel_attention_mask"),
            self._tensor(batch, "target_content_features", "content_features"),
            self._tensor(batch, "target_content_feature_lengths", "content_feature_lengths"),
            self._tensor(batch, "target_speaker_features", "speaker_features"),
        )

    def _inference_data(self, batch):
        return (
            self._tensor(batch, "source_mel_attention_mask", "mel_attention_mask"),
            self._tensor(batch, "source_content_features", "content_features"),
            self._tensor(batch, "source_content_feature_lengths", "content_feature_lengths"),
            self._tensor(batch, "reference_speaker_features", "speaker_features"),
        )

    def forward(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    def model_step(self, batch, generator=None, return_diagnostics=False):
        values = self._training_data(batch)
        kwargs = {}
        if getattr(self.method, "requires_speaker_ids", False):
            kwargs["speaker_ids"] = batch.get("target_speaker_ids", batch.get("speaker_ids"))
        if return_diagnostics and self.method.supports_diagnostics:
            return self.method.compute_loss(
                *values, generator=generator, return_diagnostics=True, **kwargs
            )
        result = (
            self.method.compute_loss(*values, **kwargs)
            if generator is None
            else self.method.compute_loss(*values, generator=generator, **kwargs)
        )
        if return_diagnostics:
            return (*result, {})
        return result

    def on_train_start(self):
        self.val_loss.reset()

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.method.supports_diagnostics:
            self.optimization_loss(loss)
            self.log(
                "train/optimization_loss",
                self.optimization_loss,
                on_step=False,
                on_epoch=True,
            )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        mask, content, lengths, speaker = self._inference_data(batch)
        kwargs = {}
        if self.method.requires_source_mels:
            kwargs["source_mels"] = self._tensor(
                batch, "source_mel_spectrograms", "mel_spectrograms"
            )
        return self.method.generate(content, lengths, speaker, mask, **kwargs)

    def _objectives(self, batch, batch_idx, sampling_count, seed_offset):
        """在固定采样下计算损失并收集诊断标量（NaN 组不并入均值）。"""
        target_mels = self._training_data(batch)[0]
        if sampling_count is None:
            loss, _, _, _, diagnostics = self.model_step(batch, return_diagnostics=True)
            return loss, diagnostics
        losses = []
        collected: dict[str, list] = {}
        for sample_idx in range(sampling_count):
            generator = torch.Generator(device=target_mels.device)
            generator.manual_seed(
                self.evaluation_seed + seed_offset + batch_idx * sampling_count + sample_idx
            )
            loss, _, _, _, diagnostics = self.model_step(
                batch, generator=generator, return_diagnostics=True
            )
            losses.append(loss)
            for name, value in diagnostics.items():
                collected.setdefault(name, []).append(value)
        mean_diagnostics = {}
        for name, values in collected.items():
            finite = [value for value in values if torch.isfinite(value)]
            if finite:
                mean_diagnostics[name] = torch.stack(finite).mean()
        return torch.stack(losses).mean(), mean_diagnostics

    def _evaluation_loss(self, batch, batch_idx, sampling_count, seed_offset):
        return self._objectives(batch, batch_idx, sampling_count, seed_offset)[0]

    def _log_validation_diagnostics(self, diagnostics, batch, batch_idx):
        """记录 B2 固定采样诊断与一步自重建 L1。"""
        for name in (
            "raw_meanflow_mse",
            "velocity_mse",
            "nonzero_interval_mse",
            "jvp_correction_rms",
        ):
            value = diagnostics.get(name)
            if value is not None and torch.isfinite(value):
                getattr(self, name)(value)
            self.log(f"val/{name}", getattr(self, name), on_step=False, on_epoch=True)
        reconstruction = self._one_step_reconstruction_l1(batch, batch_idx)
        if torch.isfinite(reconstruction):
            self.one_step_reconstruction_l1(reconstruction)
        self.log(
            "val/one_step_reconstruction_l1",
            self.one_step_reconstruction_l1,
            on_step=False,
            on_epoch=True,
        )

    def _one_step_reconstruction_l1(self, batch, batch_idx):
        """固定验证样本的一步自重建 L1；固定种子以保证可复现。"""
        target_mels, target_mask, content, lengths, speaker = self._training_data(batch)
        devices = [] if target_mels.device.type == "cpu" else [target_mels.device]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.evaluation_seed + batch_idx)
            generated = self.method.generate(content, lengths, speaker, target_mask)
        valid = target_mask.unsqueeze(1).to(dtype=target_mels.dtype)
        return ((generated - target_mels).abs() * valid).sum() / (
            valid.sum() * target_mels.size(1)
        ).clamp_min(1.0)

    def validation_step(self, batch, batch_idx):
        loss, diagnostics = self._objectives(batch, batch_idx, self.validation_sampling_count, 0)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.method.supports_diagnostics:
            self._log_validation_diagnostics(diagnostics, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        loss = (
            self.model_step(batch)[0]
            if self.test_sampling_count is None
            else self._evaluation_loss(
                batch, batch_idx, self.test_sampling_count, self._TEST_SEED_OFFSET
            )
        )
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage):
        if self.compile_model and stage == "fit":
            self.method.network = torch.compile(self.method.network)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer_factory(params=self.parameters())
        if self.warmup_steps is not None:
            scheduler = WarmupCosineLR(
                optimizer, int(self.trainer.estimated_stepping_batches), self.warmup_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @staticmethod
    def _canonical_key(key):
        if key.startswith("net."):
            return "method.network." + key[len("net.") :]
        if key.startswith("condition_encoder."):
            return "method.condition_encoder." + key[len("condition_encoder.") :]
        return key

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped: Any = OrderedDict()
        for key, value in state_dict.items():
            destination = self._canonical_key(key)
            if destination in mapped:
                raise ValueError(f"检查点键冲突: {destination}")
            mapped[destination] = value
        if hasattr(state_dict, "_metadata"):
            mapped._metadata = OrderedDict(
                (self._canonical_key(key + ".").rstrip("."), value)
                for key, value in getattr(state_dict, "_metadata").items()
            )
        return super().load_state_dict(mapped, strict=strict, assign=assign)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["voice_parameter_names"] = [name for name, _ in self.named_parameters()]

    def on_load_checkpoint(self, checkpoint):
        # 权重可以映射，旧优化器却缺少可验证的参数名与分组对应关系。
        if checkpoint.get("optimizer_states") and getattr(self, "_trainer", None) is not None:
            saved_names = checkpoint.get("voice_parameter_names")
            names = [name for name, _ in self.named_parameters()]
            if saved_names != names:
                raise RuntimeError(
                    "不能验证旧检查点的完整 optimizer 恢复；请先 load_from_checkpoint "
                    "或严格 load_state_dict 加载权重，再不带 ckpt_path 启动训练。"
                )
