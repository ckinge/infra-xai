"""
实验2: 多粒度解释质量评估
验证节点级、边级、子图级三层解释的一致性和互补性
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from data.dataset_loader import DataPipeline
from models.bilgr import BILGR
from explainability.explainer_factory import ExplainabilityFactory
from explainability.multi_granular import MultiGranularExplainer
from explainability.topology_fusion import FusionEngine, TopologyAnalyzer
import json


def adaptive_top_k(mask, fraction=0.05, min_k=3, max_k=50):
    """
    根据mask分布自适应选择top-k
    取fraction比例的非零权重边，但限制在[min_k, max_k]范围内
    """
    if mask is None or len(mask) == 0:
        return set()

    abs_mask = mask.abs()
    threshold = abs_mask.max().item() * 0.01  # 1% of max as minimum threshold
    significant = (abs_mask > threshold).sum().item()

    k = max(min_k, min(max_k, int(significant * fraction), significant))
    if k == 0:
        return set()

    _, indices = torch.topk(abs_mask, k)
    return set(indices.tolist())


def run_experiment_2():
    """多粒度解释质量评估"""
    print("=" * 60)
    print("Experiment 2: Multi-Granular Explanation Quality")
    print("=" * 60)

    device = cfg.experiment.device
    results = {'networks': {}}

    for net_type in ['scale_free', 'small_world', 'random']:
        print(f"\n--- Network: {net_type} ---")

        # 1. 准备数据和模型
        pipeline = DataPipeline(cfg)
        data = pipeline.run(
            "synthetic", n_nodes=500, network_type=net_type,
        ).to(device)

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
            loss = torch.nn.functional.cross_entropy(
                out[data.train_mask], data.y[data.train_mask]
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=-1)
            acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f"  Model test accuracy: {acc:.4f}")

        # 2. 构建解释器
        factory = ExplainabilityFactory(model, device, cfg.explain)
        gnnexplainer = factory.build_gnnexplainer()
        multi = MultiGranularExplainer(model, gnnexplainer, cfg.explain)

        # 3. 对多个测试节点进行多粒度分析
        test_nodes = torch.where(data.test_mask)[0][:20]
        network_results = {
            'node_edge_consistency': [],
            'edge_subgraph_consistency': [],
            'node_subgraph_consistency': [],
            'num_important_edges': [],
            'subgraph_sizes': [],
        }

        for node_idx in test_nodes.tolist():
            try:
                report = multi.generate_full_report(
                    data, node_idx,
                    top_k_nodes=20, top_k_edges=30, subgraph_radius=2,
                )

                # 自适应获取重要集合
                edge_mask = report['edge_level']['edge_mask']

                # 重要边集合（自适应阈值）
                important_edges = adaptive_top_k(edge_mask, fraction=0.05)
                if len(important_edges) == 0:
                    continue

                # 重要节点：重要边涉及的所有节点
                important_nodes = set()
                for e_idx in important_edges:
                    important_nodes.add(data.edge_index[0, e_idx].item())
                    important_nodes.add(data.edge_index[1, e_idx].item())

                # 脆弱节点（来自节点级排名）
                vuln_nodes = set(report['node_level']['vulnerable_nodes'][:15])

                # 子图节点
                sub_nodes = set(report['subgraph_level']['subgraph_nodes'])

                # 一致性计算
                # 节点-边一致性
                node_edge = len(important_nodes & vuln_nodes) / max(
                    len(important_nodes | vuln_nodes), 1
                )

                # 边-子图一致性
                edge_sub = len(important_nodes & sub_nodes) / max(
                    len(important_nodes | sub_nodes), 1
                )

                # 节点-子图一致性
                node_sub = len(vuln_nodes & sub_nodes) / max(
                    len(vuln_nodes | sub_nodes), 1
                )

                network_results['node_edge_consistency'].append(node_edge)
                network_results['edge_subgraph_consistency'].append(edge_sub)
                network_results['node_subgraph_consistency'].append(node_sub)
                network_results['num_important_edges'].append(len(important_edges))
                network_results['subgraph_sizes'].append(len(sub_nodes))

            except Exception as e:
                print(f"    Warning: node {node_idx} failed: {e}")

        # 汇总该网络的指标
        for key in network_results:
            vals = network_results[key]
            if vals:
                results['networks'][net_type] = results['networks'].get(net_type, {})
                if key.startswith('num_') or key.startswith('subgraph'):
                    results['networks'][net_type][key] = float(np.mean(vals))
                else:
                    results['networks'][net_type][key] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                    }

        # 打印
        if network_results['node_edge_consistency']:
            print(f"  Node-Edge consistency:    {np.mean(network_results['node_edge_consistency']):.3f}")
            print(f"  Edge-Subgraph consistency: {np.mean(network_results['edge_subgraph_consistency']):.3f}")
            print(f"  Node-Subgraph consistency: {np.mean(network_results['node_subgraph_consistency']):.3f}")
            print(f"  Avg important edges:      {np.mean(network_results['num_important_edges']):.1f}")
            print(f"  Avg subgraph size:        {np.mean(network_results['subgraph_sizes']):.1f}")

    # 4. 拓扑-学习融合分析
    print("\n--- Topology-Learning Fusion ---")
    pipeline = DataPipeline(cfg)
    data_fusion = pipeline.run("synthetic", n_nodes=500, network_type="scale_free").to(device)
    G = data_fusion.nx_graph

    model_fusion = BILGR(
        in_channels=cfg.model.in_channels,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        num_layers=cfg.model.num_layers,
    ).to(device)

    opt = torch.optim.Adam(model_fusion.parameters(), lr=cfg.experiment.lr)
    model_fusion.train()
    for epoch in range(cfg.experiment.epochs):
        opt.zero_grad()
        out = model_fusion(data_fusion.x, data_fusion.edge_index)
        loss = torch.nn.functional.cross_entropy(
            out[data_fusion.train_mask], data_fusion.y[data_fusion.train_mask]
        )
        loss.backward()
        opt.step()

    model_fusion.eval()
    with torch.no_grad():
        logits = model_fusion(data_fusion.x, data_fusion.edge_index)
        learned_scores = logits.softmax(dim=-1)[:, 0]

    topo_metrics = TopologyAnalyzer.compute_all_metrics(G)
    fusion_engine = FusionEngine(
        topo_weight=cfg.explain.topo_weight,
        learn_weight=cfg.explain.learn_weight,
    )

    # 多k值的agreement
    for k in [5, 10, 20, 50]:
        agreement = fusion_engine.compute_fusion_agreement(
            learned_scores, topo_metrics, top_k=k
        )
        print(f"  Top-{k} agreement: {agreement:.3f}")

    # 保存
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.experiment.output_dir, "exp2_multigranular.json")

    serializable = {'networks': {}}
    for net, vals in results['networks'].items():
        serializable['networks'][net] = {}
        for k, v in vals.items():
            if isinstance(v, dict):
                serializable['networks'][net][k] = v
            else:
                serializable['networks'][net][k] = v

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_2()
