from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]
experiment_dir = Path("configs/experiment")
experiment_names = sorted(path.stem for path in experiment_dir.glob("*.yaml"))


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.parametrize("experiment_name", experiment_names)
def test_experiments(
    tmp_path: Path, experiment_name: str, experiment_data_overrides: list[str]
) -> None:
    """使用快速开发模式运行单个实验配置，验证配置能够完成一次训练流程。

    :param tmp_path: 当前实验保存 Hydra sweep 输出的临时目录。
    :param experiment_name: ``configs/experiment`` 下的实验配置名称。
    :param experiment_data_overrides: 真实或临时实验数据对应的 Hydra 覆盖参数。
    :return: 无返回值；训练命令失败时测试直接失败。
    """
    command = (
        [
            startfile,
            "-m",
            "experiment=" + experiment_name,
            "hydra.sweep.dir=" + str(tmp_path / experiment_name),
            "++trainer.fast_dev_run=true",
            "data.batch_size=2",
        ]
        + experiment_data_overrides
        + overrides
    )
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path) -> None:
    """使用两个学习率运行默认 Hydra sweep，验证多任务调度功能。

    :param tmp_path: 保存 sweep 输出的临时目录。
    :return: 无返回值；任一 sweep 任务失败时测试直接失败。
    """
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path: Path) -> None:
    """使用模拟 DDP 和三个学习率运行 Hydra sweep。

    :param tmp_path: 保存模拟分布式 sweep 输出的临时目录。
    :return: 无返回值；任一训练任务失败时测试直接失败。
    """
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path: Path) -> None:
    """运行 Optuna 超参数搜索，验证采样器和 Hydra sweeper 集成。

    :param tmp_path: 保存 Optuna 搜索输出的临时目录。
    :return: 无返回值；超参数搜索命令失败时测试直接失败。
    """
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(wandb=True, sh=True)
@pytest.mark.slow
def test_optuna_sweep_ddp_sim_wandb(tmp_path: Path) -> None:
    """使用模拟 DDP 和 W&B 日志运行 Optuna 超参数搜索。

    :param tmp_path: 保存 Optuna 搜索输出的临时目录。
    :return: 无返回值；搜索或日志初始化失败时测试直接失败。
    """
    command = [
        startfile,
        "-m",
        "hparams_search=mnist_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "logger=wandb",
    ]
    run_sh_command(command)
