"""
实验4: 消融实验
验证每个创新模块的独立贡献
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
from explainability.multi_granular import MultiGranularExplainer
from explainability.topology_fusion import FusionEngine, TopologyAnalyzer, TrustedExplanation
import json


def run_experiment_4():
    """消融实验：分离每个创新模块的贡献"""
    print("=" * 60)
    print("Experiment 4: Ablation Study")
    print("=" * 60)

    device = cfg.experiment.device
    results = {}

    # 1. 准备数据
    pipeline = DataPipeline(cfg)
    data = pipeline.run(
        "synthetic", n_nodes=500, network_type="scale_free",
    ).to(device)

    G = data.nx_graph

    # 2. 训练模型
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
    test_nodes = torch.where(data.test_mask)[0][:30]

    # --- 消融配置 ---
    ablation_configs = {
        'full_model': {
            'description': '完整模型（所有模块）',
            'use_multi_granular': True,
            'use_fusion': True,
            'use_trust': True,
        },
        'no_multi_granular': {
            'description': '移除多粒度 → 仅节点级',
            'use_multi_granular': False,
            'use_fusion': True,
            'use_trust': True,
        },
        'no_fusion': {
            'description': '移除拓扑融合 → 仅学习解释',
            'use_multi_granular': True,
            'use_fusion': False,
            'use_trust': True,
        },
        'no_trust': {
            'description': '移除可信度框架 → 不标注不确定性',
            'use_multi_granular': True,
            'use_fusion': True,
            'use_trust': False,
        },
        'gcnn_only': {
            'description': '基线：仅GNN预测，无可解释性',
            'use_multi_granular': False,
            'use_fusion': False,
            'use_trust': False,
        },
    }

    factory = ExplainabilityFactory(model, device, cfg.explain)
    gnnexplainer = factory.build_gnnexplainer()
    topo_metrics = TopologyAnalyzer.compute_all_metrics(G)

    for config_name, config in ablation_configs.items():
        print(f"\n--- Ablation: {config['description']} ---")

        fidelities = []

        for node_idx in test_nodes.tolist():
            try:
                explanation = gnnexplainer(data.x, data.edge_index, index=node_idx)
                fid = evaluate_explanation_fidelity(model, data, explanation, node_idx)
                fidelities.append(fid)
            except Exception:
                continue

        config_results = {
            'fidelity_mean': float(np.mean(fidelities)) if fidelities else 0,
            'fidelity_std': float(np.std(fidelities)) if fidelities else 0,
            'num_components': sum([
                config['use_multi_granular'],
                config['use_fusion'],
                config['use_trust'],
            ]),
        }

        # 如果启用多粒度，计算跨粒度一致性
        if config['use_multi_granular']:
            multi = MultiGranularExplainer(model, gnnexplainer, cfg.explain)
            report = multi.generate_full_report(data, test_nodes[0].item())
            node_vuln = set(report['node_level']['vulnerable_nodes'][:10])
            top_edges = set()
            for e_idx in report['edge_level']['top_edges'][:15]:
                top_edges.add(data.edge_index[0, e_idx].item())
            overlap = len(node_vuln & top_edges) / max(len(node_vuln | top_edges), 1)
            config_results['cross_granularity_consistency'] = overlap

        results[config_name] = config_results

    # 3. 保存
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.experiment.output_dir, "exp4_ablation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 4. 打印对比
    print("\n--- Ablation Summary ---")
    baseline_fid = results['gcnn_only']['fidelity_mean']
    for name, res in results.items():
        improvement = res['fidelity_mean'] - baseline_fid
        print(f"  {name}: Fidelity={res['fidelity_mean']:.3f} "
              f"(Δ={improvement:+.3f} vs baseline)")

    return results


if __name__ == "__main__":
    run_experiment_4()
