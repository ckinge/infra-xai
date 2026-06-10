"""
一键运行所有实验
用法: python experiments/run_all.py
可选: python experiments/run_all.py --exp 1,2,3  # 只运行指定实验
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import cfg


def run_all(exp_ids=None):
    """按顺序运行所有实验"""
    experiments = {
        1: ("可解释性方法基准比较", "exp1_baseline.run_experiment_1"),
        2: ("多粒度解释质量评估", "exp2_multigranular.run_experiment_2"),
        3: ("反事实韧性提升验证", "exp3_counterfactual.run_experiment_3"),
        4: ("消融实验", "exp4_ablation.run_experiment_4"),
        5: ("鲁棒性分析", "exp5_robustness.run_experiment_5"),
    }

    if exp_ids is None:
        exp_ids = list(experiments.keys())

    print("=" * 60)
    print("Infra-XAI: Full Experiment Pipeline")
    print(f"Device: {cfg.experiment.device}")
    print(f"Output: {cfg.experiment.output_dir}")
    print("=" * 60)

    for exp_id in sorted(exp_ids):
        if exp_id not in experiments:
            print(f"Warning: Experiment {exp_id} not found, skipping")
            continue

        name, import_path = experiments[exp_id]
        print(f"\n{'#' * 60}")
        print(f"# Experiment {exp_id}: {name}")
        print(f"{'#' * 60}")

        try:
            module_name, func_name = import_path.rsplit('.', 1)
            module = __import__(
                f"experiments.{module_name}", fromlist=[func_name]
            )
            func = getattr(module, func_name)
            func()
        except Exception as e:
            print(f"ERROR in Experiment {exp_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Infra-XAI experiments")
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Comma-separated experiment IDs to run (e.g., '1,2,3')"
    )
    args = parser.parse_args()

    exp_ids = None
    if args.exp:
        exp_ids = [int(x.strip()) for x in args.exp.split(",")]

    run_all(exp_ids)
