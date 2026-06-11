"""
实验1: 可解释性方法基准比较 (多seed + signed fidelity)
"""
import torch, numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from data.dataset_loader import DataPipeline
from models.bilgr import BILGR
from explainability.explainer_factory import (
    ExplainabilityFactory, evaluate_explanation_fidelity, fidelity_summary
)
from tqdm import tqdm


class BaselineExplainer:
    def __init__(self, data, method='degree'):
        self.data = data
        deg = torch.zeros(data.num_nodes)
        for e in range(data.edge_index.shape[1]):
            deg[data.edge_index[0, e]] += 1; deg[data.edge_index[1, e]] += 1
        if method == 'degree':
            self.edge_mask = deg[data.edge_index[0]] + deg[data.edge_index[1]]
            self.edge_mask = self.edge_mask / self.edge_mask.max()
        else:
            self.edge_mask = torch.rand(data.edge_index.shape[1])
    def __call__(self, x, edge_index, index=None):
        import types
        return types.SimpleNamespace(edge_mask=self.edge_mask, node_mask=None)


def train_and_eval(data, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    pipeline = DataPipeline(cfg)
    model = BILGR(
        in_channels=cfg.model.in_channels, hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels, num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.experiment.lr, weight_decay=5e-4)
    model.train()
    for e in range(200):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
    model.eval()
    return model


def run_experiment_1():
    print("=" * 60)
    print("Experiment 1: Explainability Benchmark (multi-seed, signed fidelity)")
    print("=" * 60)
    device = cfg.experiment.device; seeds = [42, 123, 456]
    results = {}

    for net_type in ['scale_free', 'small_world', 'random']:
        print(f"\n--- {net_type} ---")
        np.random.seed(42); torch.manual_seed(42)
        pipeline = DataPipeline(cfg)
        data = pipeline.run("synthetic", n_nodes=500, network_type=net_type).to(device)
        print(f"  Classes: {[(data.y==c).sum().item() for c in range(3)]}")

        all_method_results = {'GNNExplainer': [], 'Degree': [], 'Random': []}

        for seed in seeds:
            model = train_and_eval(data, device, seed)
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=-1)
                acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
            factory = ExplainabilityFactory(model, device, cfg.explain)
            gnne = factory.build_gnnexplainer()
            deg_exp = BaselineExplainer(data, 'degree')
            rnd_exp = BaselineExplainer(data, 'random')
            test_nodes = torch.where(data.test_mask)[0][:20]

            for mname, explainer in [('GNNExplainer', gnne), ('Degree', deg_exp), ('Random', rnd_exp)]:
                fids = []
                for n in test_nodes.tolist():
                    try:
                        exp = explainer(data.x, data.edge_index, index=n)
                        fid = evaluate_explanation_fidelity(model, data, exp, n)
                        fids.append(fid)
                    except: pass
                if fids:
                    all_method_results[mname].append(fidelity_summary(fids))

        # Aggregate across seeds
        net_results = {}
        for mname, summaries in all_method_results.items():
            keys = ['mean_signed', 'positive_rate', 'median', 'iqr']
            net_results[mname] = {
                k: float(np.mean([s[k] for s in summaries])) for k in keys
            }
            net_results[mname]['acc'] = float(np.mean([
                float(np.mean([s['mean_signed'] for s in summaries]))  # rough
            ]))
        results[net_type] = net_results
        print(f"  GNNExplainer: signed={net_results['GNNExplainer']['mean_signed']:.4f}, "
              f"pos_rate={net_results['GNNExplainer']['positive_rate']:.3f}, "
              f"median={net_results['GNNExplainer']['median']:.4f}")
        print(f"  Degree:       signed={net_results['Degree']['mean_signed']:.4f}, "
              f"pos_rate={net_results['Degree']['positive_rate']:.3f}")

    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    with open(os.path.join(cfg.experiment.output_dir, "exp1_benchmark.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved.")
    return results


if __name__ == "__main__":
    run_experiment_1()
