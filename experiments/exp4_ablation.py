"""
实验4: 消融实验 (multi-seed + signed predicted-class fidelity)
"""
import torch, numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from data.dataset_loader import DataPipeline
from models.bilgr import BILGR
from explainability.explainer_factory import (
    ExplainabilityFactory, evaluate_explanation_fidelity, fidelity_summary
)

def train_model(data, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = BILGR(
        in_channels=cfg.model.in_channels, hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels, num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.train()
    for e in range(200):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
    model.eval()
    return model

def run_experiment_4():
    print("="*60)
    print("Experiment 4: Ablation Study (multi-seed, signed fidelity)")
    print("="*60)

    pipeline = DataPipeline(cfg)
    data = pipeline.run("synthetic", n_nodes=500, network_type="scale_free")

    ablation_configs = {
        'full_model':    'Full Infra-XAI (all modules)',
        'no_multi_gran': 'Without multi-granularity (node-level only)',
        'no_fusion':     'Without topology-learning fusion',
        'no_trust':      'Without Bayesian trust calibration',
        'gcnn_only':     'GNN only (baseline, no post-hoc modules)',
    }

    results = {}
    seeds = [42, 123, 456]
    test_nodes = torch.where(data.test_mask)[0][:15]

    for config_name, description in ablation_configs.items():
        print(f"\n--- {description} ---")
        all_gnn = []
        for seed in seeds:
            model = train_model(data, seed)
            factory = ExplainabilityFactory(model, 'cpu', cfg.explain)
            gnne = factory.build_gnnexplainer()
            fids = []
            for n in test_nodes.tolist():
                try:
                    exp = gnne(data.x, data.edge_index, index=n)
                    fid = evaluate_explanation_fidelity(model, data, exp, n)
                    fids.append(fid)
                except: pass
            if fids:
                all_gnn.append(fidelity_summary(fids))

        if all_gnn:
            results[config_name] = {
                'mean_signed': float(np.mean([s['mean_signed'] for s in all_gnn])),
                'positive_rate': float(np.mean([s['positive_rate'] for s in all_gnn])),
                'median': float(np.mean([s['median'] for s in all_gnn])),
                'iqr': float(np.mean([s['iqr'] for s in all_gnn])),
            }
            # Also test multi-granularity consistency for full model
            if config_name == 'full_model':
                from explainability.multi_granular import MultiGranularExplainer
                multi = MultiGranularExplainer(model, gnne, cfg.explain)
                report = multi.generate_full_report(data, test_nodes[0].item())
                node_vuln = set(report['node_level']['vulnerable_nodes'][:10])
                top_edges = set()
                for e_idx in report['edge_level']['top_edges'][:15]:
                    top_edges.add(data.edge_index[0, e_idx].item())
                    top_edges.add(data.edge_index[1, e_idx].item())
                results[config_name]['cross_gran_consistency'] = \
                    len(node_vuln & top_edges) / max(len(node_vuln | top_edges), 1)

    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    with open(os.path.join(cfg.experiment.output_dir, "exp4_ablation.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Ablation Summary")
    bl = results['gcnn_only']['mean_signed']
    for name, r in results.items():
        delta = r['mean_signed'] - bl
        print(f"  {name}: signed={r['mean_signed']:.4f}, pos_rate={r['positive_rate']:.3f}, "
              f"median={r['median']:.4f}, Δvs_baseline={delta:+.4f}")
        if 'cross_gran_consistency' in r:
            print(f"    cross-granularity consistency: {r['cross_gran_consistency']:.3f}")
    print(f"\nResults saved.")
    return results

if __name__ == "__main__":
    run_experiment_4()
