from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_b1_b2_controlled_defaults() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        baseline = hydra.compose(
            config_name="train.yaml", overrides=["experiment=conditional_flow_matching_vctk"]
        )
        mean_flow = hydra.compose(
            config_name="train.yaml", overrides=["experiment=conditional_mean_flow_vctk"]
        )
        mean_voice_flow = hydra.compose(
            config_name="train.yaml", overrides=["experiment=meanvoiceflow_vctk"]
        )

    for field in ("seed", "data", "callbacks", "logger"):
        assert baseline[field] == mean_flow[field]
        assert baseline[field] == mean_voice_flow[field]
    for field in (
        "optimizer",
        "scheduler",
        "warmup_steps",
        "compile",
        "validation_sampling_count",
        "test_sampling_count",
        "evaluation_seed",
    ):
        assert baseline.model[field] == mean_flow.model[field]
    for field in (
        "optimizer",
        "scheduler",
        "warmup_steps",
        "validation_sampling_count",
        "test_sampling_count",
        "evaluation_seed",
    ):
        assert baseline.model[field] == mean_voice_flow.model[field]
    for config in (baseline, mean_flow, mean_voice_flow):
        assert config.data.batch_size == 32
        assert config.trainer.max_epochs == 500
        assert config.callbacks.early_stopping is None
    assert baseline.model.optimizer.lr == 0.0002
    assert baseline.model.optimizer.betas == [0.5, 0.9]
    assert baseline.model.warmup_steps == 10000
    assert baseline.model.validation_sampling_count == 4
    assert baseline.model.test_sampling_count == 8
    assert baseline.model.method.condition_encoder == mean_flow.model.method.condition_encoder
    baseline_network = OmegaConf.to_container(baseline.model.method.network)
    mean_flow_network = OmegaConf.to_container(mean_flow.model.method.network)
    assert baseline_network.pop("time_fields") == ["time"]
    assert mean_flow_network.pop("time_fields") == ["time", "interval"]
    assert baseline_network == mean_flow_network
    baseline_trainer = OmegaConf.to_container(baseline.trainer)
    mean_flow_trainer = OmegaConf.to_container(mean_flow.trainer)
    assert mean_flow_trainer.pop("inference_mode") is False
    baseline_trainer.pop("inference_mode", None)
    assert baseline_trainer == mean_flow_trainer
    assert baseline.model.method.integration_steps == 50
    assert mean_flow.model.method.adaptive_loss_power == 0.0
    assert mean_flow.model.method.equal_time_probability == 0.75
