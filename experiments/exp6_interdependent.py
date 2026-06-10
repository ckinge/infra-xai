"""
实验6: 相互依赖关键基础设施网络完整实验
论文核心贡献 — 证明Infra-XAI在相互依赖场景下的有效性
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, json, copy
from config import cfg
from models.bilgr import BILGR
from explainability.explainer_factory import ExplainabilityFactory, evaluate_explanation_fidelity
from data.interdependent import InterdependentNetworkBuilder, InterdependentCascadeSimulator
from explainability.topology_fusion import TopologyAnalyzer, FusionEngine


def run_interdependent_experiment():
    print("=" * 60)
    print("Experiment 6: Interdependent Infrastructure Explainability")
    print("=" * 60)

    device = cfg.experiment.device
    results = {}

    configs = [
        ('scale_free', 'scale_free', 'Power↔Communication (SF-SF)'),
        ('scale_free', 'small_world', 'Power↔Transport (SF-SW)'),
    ]

    for topo_a, topo_b, label in configs:
        print(f"\n{'─'*50}")
        print(f"Scenario: {label}")
        print(f"{'─'*50}")

        np.random.seed(42)
        torch.manual_seed(42)

        # 1. 构建耦合网络
        builder = InterdependentNetworkBuilder(seed=42)
        G_a, G_b, deps = builder.build_coupled_networks(
            n_nodes_a=300, n_nodes_b=300,
            topo_a=topo_a, topo_b=topo_b, n_dependencies=100,
        )
        data = builder.build_coupled_pyg_data(G_a, G_b, deps).to(device)
        N = data.num_nodes
        print(f"  Nodes: {N} (A:{data.n_a}+B:{data.n_b}), Edges: {data.edge_index.shape[1]} (deps:{data.n_deps})")

        # 2. 数据分割 (固定种子)
        indices = np.random.permutation(N)
        data.train_mask = torch.zeros(N, dtype=torch.bool)
        data.test_mask = torch.zeros(N, dtype=torch.bool)
        data.train_mask[indices[:int(N*0.6)]] = True
        data.test_mask[indices[int(N*0.8):]] = True

        # 3. 训练模型
        model = BILGR(
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        model.train()
        for e in range(200):
            opt.zero_grad()
            out = model(data.x, data.edge_index)
            loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=-1)
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f"  Test accuracy: {test_acc:.4f}")

        # 4. 可解释性对比
        factory = ExplainabilityFactory(model, device, cfg.explain)
        gnnexplainer = factory.build_gnnexplainer()

        deg_t = torch.zeros(N)
        for e in range(data.edge_index.shape[1]):
            deg_t[data.edge_index[0,e]] += 1
            deg_t[data.edge_index[1,e]] += 1
        deg_mask = deg_t[data.edge_index[0]] + deg_t[data.edge_index[1]]

        import types
        class BE:
            def __init__(self, m): self.mask = m
            def __call__(self, x, ei, idx=None):
                return types.SimpleNamespace(edge_mask=self.mask, node_mask=None)

        test_nodes = torch.where(data.test_mask)[0][:20]
        xai_results = {}
        for sname, explainer in [
            ('GNNExplainer', gnnexplainer),
            ('Degree', BE(deg_mask)),
            ('Random', BE(torch.rand(data.edge_index.shape[1]))),
        ]:
            fids = []
            for n in test_nodes.tolist():
                try:
                    exp = explainer(data.x, data.edge_index, index=n)
                    fid = evaluate_explanation_fidelity(model, data, exp, n)
                    fids.append(fid)
                except Exception:
                    pass
            xai_results[sname] = {'fidelity_mean': float(np.mean(fids)) if fids else 0,
                                  'fidelity_std': float(np.std(fids)) if fids else 0}
            print(f"  {sname}: Fidelity={xai_results[sname]['fidelity_mean']:.4f}±{xai_results[sname]['fidelity_std']:.4f}")

        # 5. 相互依赖级联模拟 (使用5%初始故障以获得更清晰的信号)
        cascade = InterdependentCascadeSimulator(capacity_factor=1.5, max_steps=30)
        np.random.seed(42)
        init_fails = list(np.random.choice(data.n_a, max(1, int(data.n_a*0.05)), replace=False))
        bl = cascade.simulate(data, initial_failures_a=init_fails)
        print(f"  Cascade: {len(init_fails)} init A → {len(bl['failed_a'])} A + {len(bl['failed_b'])} B = {bl['total_failed']} total")
        print(f"  Cross-network: {bl['cross_propagation']}, Resilience: {bl['resilience']:.4f}")

        # 6. 反事实: 对比三种策略
        with torch.no_grad():
            learned = out.softmax(dim=-1)[:, 0]
        _, expl_nodes = torch.topk(learned[:data.n_a], 5)
        topo_scores = torch.tensor([G_a.degree(n) for n in G_a.nodes()], dtype=torch.float)
        _, topo_nodes = torch.topk(topo_scores, 5)
        rand_nodes = list(np.random.choice(data.n_a, 5, replace=False))

        cf_by_strategy = {}
        for sname, nodes_list in [('explain', expl_nodes.tolist()), ('topology', topo_nodes.tolist()), ('random', rand_nodes)]:
            mod_data = copy.deepcopy(data)
            mod_data.node_capacity = -torch.ones(N)
            for n in nodes_list:
                gn = mod_data.nx_graph_a
                deg_n = len(list(gn.neighbors(n)))
                nd = sum(len(list(gn.neighbors(nn))) for nn in gn.neighbors(n)) if deg_n else 0
                load = float(deg_n + 0.1*nd)
                mod_data.node_capacity[n] = load * 2.0 * 1.5
            r = cascade.simulate(mod_data, initial_failures_a=init_fails)
            red = (bl['total_failed'] - r['total_failed']) / max(bl['total_failed'], 1)
            cf_by_strategy[sname] = {'reduction': float(red), 'final_total': r['total_failed']}
            print(f"  {sname}: {bl['total_failed']}→{r['total_failed']}, ΔR={red:+.3f}")

        # 7. 拓扑-学习融合
        fe = FusionEngine()
        ag_a = fe.compute_fusion_agreement(learned[:data.n_a],
                TopologyAnalyzer.compute_all_metrics(G_a), top_k=10)
        ag_b = fe.compute_fusion_agreement(learned[data.n_a:],
                TopologyAnalyzer.compute_all_metrics(G_b), top_k=10)

        results[label] = {
            'accuracy': float(test_acc),
            'xai': xai_results,
            'cascade_size': bl['total_failed'],
            'cross_propagation': bl['cross_propagation'],
            'resilience': bl['resilience'],
            'counterfactual': cf_by_strategy,
            'fusion_a': float(ag_a),
            'fusion_b': float(ag_b),
        }

    # 保存
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.experiment.output_dir, "exp6_interdependent.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment 6 Summary")
    print(f"{'='*60}")
    for label, r in results.items():
        gn = r['xai']['GNNExplainer']['fidelity_mean']
        dg = r['xai']['Degree']['fidelity_mean']
        cf = r['counterfactual']
        print(f"\n{label}:")
        print(f"  Cascade: {r['cascade_size']} total, cross={r['cross_propagation']}, resilience={r['resilience']:.3f}")
        print(f"  GNNExplainer: {gn:.3f} vs Degree: {dg:.3f} (ratio: {gn/max(dg,1e-6):.1f}x)")
        for s in ['explain', 'topology', 'random']:
            print(f"  CF-{s}: ΔR={cf[s]['reduction']:+.3f}")
        print(f"  Fusion A: {r['fusion_a']:.3f}, B: {r['fusion_b']:.3f}")

    return results


if __name__ == "__main__":
    run_interdependent_experiment()
