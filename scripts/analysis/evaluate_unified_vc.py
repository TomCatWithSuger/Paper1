"""运行固定 100 组 source/reference 的统一 VC 评估。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.unified_vc_evaluation import evaluate_unified  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一评估 B1/B2/B3 的 Mel 质量与效率")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("analysis_results/baseline_validation/experiment_plan.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("analysis_results/baseline_validation/test_manifest.csv"),
    )
    parser.add_argument("--vctk-root", type=Path, required=True)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("data/meanvc_features/vctk"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/unified_evaluation"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--experiment-id", action="append", dest="experiment_ids")
    return parser


def main() -> None:
    """运行统一 VC 生成与指标评估。"""
    args = _parser().parse_args()
    outputs = evaluate_unified(
        plan_path=args.plan,
        manifest_path=args.manifest,
        vctk_root=args.vctk_root,
        feature_cache_dir=args.feature_cache_dir,
        output_dir=args.output_dir,
        device=args.device,
        base_seed=args.base_seed,
        limit=args.limit,
        experiment_ids=args.experiment_ids,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
