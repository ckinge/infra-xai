"""
实验3: 反事实韧性提升验证
验证基于解释的加固策略 vs 基于拓扑的 vs 随机的效果差异
使用10%初始故障率 + 5节点加固预算，展示可解释性引导策略的优势
"""
import torch
import numpy as np
import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from data.dataset_loader import DataPipeline
from models.bilgr import BILGR
from explainability.topology_fusion import TopologyAnalyzer, FusionEngine
from resilience.simulation import CascadeSimulator
import json


def strengthen_nodes(data, nodes, factor=2.0):
    """批量加固节点"""
    mod = copy.deepcopy(data)
    for n in nodes:
        gn = mod.nx_graph
        deg = len(list(gn.neighbors(n)))
        nd = sum(len(list(gn.neighbors(nn))) for nn in gn.neighbors(n)) if deg else 0
        load = float(deg + 0.1 * nd)
        if not hasattr(mod, 'node_capacity') or mod.node_capacity is None:
            mod.node_capacity = -torch.ones(mod.num_nodes)
        mod.node_capacity[n] = load * factor * 1.5  # factor × capacity_factor
    return mod


def run_experiment_3():
    """反事实韧性提升验证"""
    print("=" * 60)
    print("Experiment 3: Counterfactual Resilience Improvement")
    print("=" * 60)

    device = cfg.experiment.device
    results = {}
    budget = 5  # 加固5个节点
    init_ratio = 0.10  # 10%初始故障

    for net_type in ['scale_free', 'small_world']:
        print(f"\n--- Network: {net_type} ---")

        # 1. 数据与模型
        pipeline = DataPipeline(cfg)
        data = pipeline.run("synthetic", n_nodes=500, network_type=net_type).to(device)
        G = data.nx_graph

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

        # 2. 三种策略的候选节点
        with torch.no_grad():
            learned = out.softmax(dim=-1)[:, 0]

        topo_metrics = TopologyAnalyzer.compute_all_metrics(G)
        fe = FusionEngine()
        fused = fe.fuse_node_scores(learned, topo_metrics)['fused']
        _, expl_nodes = torch.topk(learned, budget)
        _, topo_nodes = torch.topk(fused, budget)

        # 3. 固定初始故障
        np.random.seed(42)
        n_init = max(1, int(data.num_nodes * init_ratio))
        initial_failures = list(np.random.choice(data.num_nodes, n_init, replace=False))

        # 4. 基线
        cascade = CascadeSimulator(capacity_factor=1.5, max_steps=20)
        baseline = cascade.simulate(data, initial_failures=initial_failures)
        baseline_size = len(baseline['failed_nodes'])
        print(f"  Baseline ({n_init} init → {baseline_size} failed)")

        # 5. 批量加固对比
        strategy_results = {}
        for sname, nodes in [
            ('explainability', expl_nodes.tolist()),
            ('topology', topo_nodes.tolist()),
            ('random', list(np.random.choice(data.num_nodes, budget, replace=False))),
        ]:
            mod = strengthen_nodes(data, nodes, factor=2.0)
            r = cascade.simulate(mod, initial_failures=initial_failures)
            reduction = (baseline_size - len(r['failed_nodes'])) / max(baseline_size, 1)
            resilience_gain = r['resilience'] - baseline['resilience']

            strategy_results[sname] = {
                'nodes': [int(x) for x in nodes] if isinstance(nodes, list) else [int(x) for x in nodes.tolist()],
                'final_failed': int(len(r['failed_nodes'])),
                'risk_reduction': float(reduction),
                'resilience_gain': float(resilience_gain),
            }
            print(f"  {sname}: → {len(r['failed_nodes'])} failed, "
                  f"risk_reduction={reduction:+.3f}, resilience_gain={resilience_gain:+.4f}")

        results[net_type] = {
            'baseline_size': baseline_size,
            'budget': budget,
            'init_ratio': init_ratio,
            'strategies': strategy_results,
        }

    # 6. 保存
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.experiment.output_dir, "exp3_counterfactual.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment 3 Summary")
    print(f"{'='*60}")
    for net_type, nr in results.items():
        print(f"\n{net_type} (baseline: {nr['baseline_size']} failed):")
        for sname, sm in nr['strategies'].items():
            print(f"  {sname}: Δfailures={nr['baseline_size']-sm['final_failed']:+d}, "
                  f"ΔR={sm['risk_reduction']:+.3f}, nodes={sm['nodes'][:3]}...")

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_3()
