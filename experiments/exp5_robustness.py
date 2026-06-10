"""
实验5: 鲁棒性分析
在不同条件下验证方法稳定性
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from data.dataset_loader import DataPipeline
from models.bilgr import BILGR
from explainability.explainer_factory import ExplainabilityFactory, evaluate_explanation_fidelity
from explainability.counterfactual import CounterfactualAnalyzer
from resilience.simulation import CascadeSimulator
import json


def run_experiment_5():
    """鲁棒性分析"""
    print("=" * 60)
    print("Experiment 5: Robustness Analysis")
    print("=" * 60)

    device = cfg.experiment.device
    results = {
        'varying_failure_ratios': {},
        'varying_network_size': {},
        'varying_topology': {},
        'varying_noise': {},
    }

    # ─── 5.1 不同初始故障规模 ────────────────────────
    print("\n--- 5.1 Varying Initial Failure Ratios ---")
    pipeline = DataPipeline(cfg)
    data = pipeline.run("synthetic", n_nodes=500, network_type="scale_free").to(device)

    model = BILGR(
        in_channels=cfg.model.in_channels,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        num_layers=cfg.model.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiment.lr)
    model.train()
    for epoch in range(cfg.experiment.epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    cascade_sim = CascadeSimulator(max_steps=cfg.resilience.max_cascade_steps)

    for ratio in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]:
        ratio_results = []
        for _ in range(cfg.experiment.num_repeats):
            sim_result = cascade_sim.simulate(data, failure_ratio=ratio)
            ratio_results.append({
                'final_failure_size': len(sim_result['failed_nodes']),
                'resilience': sim_result['resilience'],
                'cascade_steps': sim_result['total_steps'],
            })

        results['varying_failure_ratios'][str(ratio)] = {
            'mean_failure_size': float(np.mean([r['final_failure_size'] for r in ratio_results])),
            'std_failure_size': float(np.std([r['final_failure_size'] for r in ratio_results])),
            'mean_resilience': float(np.mean([r['resilience'] for r in ratio_results])),
        }

    # ─── 5.2 不同网络规模 ──────────────────────────
    print("--- 5.2 Varying Network Size ---")
    for n_nodes in [100, 200, 500, 1000]:
        size_data = pipeline.run(
            "synthetic", n_nodes=n_nodes, network_type="scale_free",
        ).to(device)

        model_small = BILGR(
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
        ).to(device)

        optimizer = torch.optim.Adam(model_small.parameters(), lr=cfg.experiment.lr)
        model_small.train()
        for epoch in range(cfg.experiment.epochs):
            optimizer.zero_grad()
            out = model_small(size_data.x, size_data.edge_index)
            loss = torch.nn.functional.cross_entropy(
                out[size_data.train_mask], size_data.y[size_data.train_mask]
            )
            loss.backward()
            optimizer.step()

        # 评估
        model_small.eval()
        factory = ExplainabilityFactory(model_small, device, cfg.explain)
        explainer = factory.build_gnnexplainer()
        test_nodes = torch.where(size_data.test_mask)[0][:20]

        fids = []
        for node_idx in test_nodes.tolist():
            try:
                explanation = explainer(size_data.x, size_data.edge_index, index=node_idx)
                fid = evaluate_explanation_fidelity(model_small, size_data, explanation, node_idx)
                fids.append(fid)
            except Exception:
                continue

        cascade_sim_small = CascadeSimulator(max_steps=cfg.resilience.max_cascade_steps)
        sim_result = cascade_sim_small.simulate(size_data, failure_ratio=0.05)

        results['varying_network_size'][str(n_nodes)] = {
            'fidelity': float(np.mean(fids)) if fids else 0,
            'resilience': sim_result['resilience'],
            'failure_size': len(sim_result['failed_nodes']),
        }

    # ─── 5.3 不同网络拓扑 ─────────────────────────
    print("--- 5.3 Varying Network Topology ---")
    for topo in ['scale_free', 'small_world', 'random', 'clustered']:
        topo_data = pipeline.run(
            "synthetic", n_nodes=500, network_type=topo,
        ).to(device)

        model_topo = BILGR(
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
        ).to(device)

        optimizer = torch.optim.Adam(model_topo.parameters(), lr=cfg.experiment.lr)
        model_topo.train()
        for epoch in range(cfg.experiment.epochs):
            optimizer.zero_grad()
            out_t = model_topo(topo_data.x, topo_data.edge_index)
            loss_t = torch.nn.functional.cross_entropy(
                out_t[topo_data.train_mask], topo_data.y[topo_data.train_mask]
            )
            loss_t.backward()
            optimizer.step()

        model_topo.eval()
        cascade_sim_topo = CascadeSimulator(max_steps=cfg.resilience.max_cascade_steps)
        sim_topo = cascade_sim_topo.simulate(topo_data, failure_ratio=0.05)

        results['varying_topology'][topo] = {
            'resilience': sim_topo['resilience'],
            'failure_size': len(sim_topo['failed_nodes']),
            'cascade_steps': sim_topo['total_steps'],
        }

    # 5.4 噪声鲁棒性
    print("--- 5.4 Robustness to Observation Noise ---")
    base_data = pipeline.run("synthetic", n_nodes=500, network_type="scale_free").to(device)

    for noise_level in [0.0, 0.05, 0.10, 0.15, 0.20]:
        noisy_data = base_data.clone()
        # 随机添加/删除边
        n_edges = noisy_data.edge_index.shape[1]
        n_noise = int(n_edges * noise_level)

        if n_noise > 0:
            # 删除随机边
            remove_indices = torch.randperm(n_edges, device=device)[:n_noise // 2]
            mask = torch.ones(n_edges, dtype=torch.bool)
            mask[remove_indices] = False
            noisy_data.edge_index = noisy_data.edge_index[:, mask]

            # 添加随机边
            new_edges = torch.randint(0, noisy_data.num_nodes, (2, n_noise // 2), device=device)
            noisy_data.edge_index = torch.cat([noisy_data.edge_index, new_edges], dim=1)

        cascade_noisy = CascadeSimulator(max_steps=cfg.resilience.max_cascade_steps)
        sim_noisy = cascade_noisy.simulate(noisy_data, failure_ratio=0.05)

        results['varying_noise'][str(noise_level)] = {
            'resilience': sim_noisy['resilience'],
            'failure_size': len(sim_noisy['failed_nodes']),
        }

    # 保存
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.experiment.output_dir, "exp5_robustness.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_5()
