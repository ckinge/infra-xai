"""
实验7: Bayesian Trust Calibration 实证验证
证明贝叶斯不确定性量化不是"概念模块"，而是有实际预测能力的
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, json
from config import cfg
from data.dataset_loader import DataPipeline
from models.bilgr import BILGR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error"""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            bin_size = in_bin.sum()
            ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)
            bin_stats.append({'center': (bin_boundaries[i]+bin_boundaries[i+1])/2,
                            'acc': float(bin_acc), 'conf': float(bin_conf), 'size': int(bin_size)})
    return ece, bin_stats


def run_calibration_experiment():
    print("=" * 60)
    print("Experiment 7: Bayesian Trust Calibration Validation")
    print("=" * 60)

    device = cfg.experiment.device
    results = {}

    for net_type in ['scale_free', 'small_world', 'random']:
        print(f"\n--- {net_type} ---")

        np.random.seed(42); torch.manual_seed(42)
        pipeline = DataPipeline(cfg)
        data = pipeline.run('synthetic', n_nodes=500, network_type=net_type).to(device)

        model = BILGR(
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            mc_samples=30,
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
        test_mask = data.test_mask
        test_labels = data.y[test_mask].cpu().numpy()

        # MC Dropout uncertainty (without MAP graph estimation —
        # pure epistemic uncertainty from weight sampling)
        logits_mean, logits_var, all_samples = model.mc_wrapper.predict(
            data.x, data.edge_index
        )
        logits_mean = logits_mean[test_mask].cpu()
        logits_var = logits_var[test_mask].cpu()
        probs = logits_mean.softmax(dim=-1).numpy()
        preds = probs.argmax(axis=1)
        correct = (preds == test_labels).astype(int)

        # Uncertainty: predictive entropy + mean variance
        entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        total_var = logits_var.sum(dim=-1).numpy()
        uncertainties = entropy + total_var
        # Normalize
        u_min, u_max = uncertainties.min(), uncertainties.max()
        if u_max > u_min:
            uncertainties = (uncertainties - u_min) / (u_max - u_min)

        # 1. ECE
        ece, bin_stats = compute_ece(probs, test_labels)
        print(f"  ECE: {ece:.4f}")
        print(f"  Accuracy: {correct.mean():.4f}")

        # 2. Brier Score (multi-class: per-class mean)
        brier_per_class = []
        for c in range(3):
            y_bin = (test_labels == c).astype(float)
            brier_per_class.append(np.mean((probs[:, c] - y_bin) ** 2))
        brier = np.mean(brier_per_class)
        print(f"  Brier Score: {brier:.4f}")

        # 3. Uncertainty vs Error correlation
        error = 1 - correct
        n_uncertain = int(len(uncertainties) * 0.3)
        high_u_idx = np.argsort(uncertainties)[-n_uncertain:]
        low_u_idx = np.argsort(uncertainties)[:n_uncertain]
        high_u_err = error[high_u_idx].mean()
        low_u_err = error[low_u_idx].mean()
        print(f"  High-uncertainty error rate: {high_u_err:.4f}")
        print(f"  Low-uncertainty error rate:  {low_u_err:.4f}")
        print(f"  Error discrimination ratio: {high_u_err/max(low_u_err,1e-6):.1f}x")

        # 4. Uncertainty vs error Pearson correlation
        u_err_corr = np.corrcoef(uncertainties, error)[0, 1]
        print(f"  Uncertainty-Error correlation: {u_err_corr:.4f}")

        # 5. Reliability Diagram
        fig, ax = plt.subplots(figsize=(6,6))
        for b in bin_stats:
            ax.plot(b['center'], b['acc'], 'o', markersize=b['size']*2, color='#2e86c1')
        ax.plot([0,1],[0,1],'--',color='gray',alpha=0.5,label='Perfect calibration')
        ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
        ax.set_title(f'Reliability Diagram: {net_type} (ECE={ece:.4f})')
        ax.legend(); ax.grid(True, alpha=0.3)
        os.makedirs(f'{cfg.experiment.output_dir}/figures', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'{cfg.experiment.output_dir}/figures/reliability_{net_type}.pdf')
        plt.close()

        results[net_type] = {
            'ece': float(ece),
            'brier': float(brier),
            'accuracy': float(correct.mean()),
            'high_uncertainty_error_rate': float(high_u_err),
            'low_uncertainty_error_rate': float(low_u_err),
            'error_discrimination_ratio': float(high_u_err / max(low_u_err, 1e-6)),
            'uncertainty_error_correlation': float(u_err_corr),
            'reliability_bins': bin_stats,
        }

    # 保存
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.experiment.output_dir, "exp7_calibration.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment 7 Summary")
    print(f"{'='*60}")
    for net_type, r in results.items():
        print(f"\n{net_type}:")
        print(f"  ECE={r['ece']:.4f}, Brier={r['brier']:.4f}")
        print(f"  High-U error: {r['high_uncertainty_error_rate']:.4f} vs Low-U: {r['low_uncertainty_error_rate']:.4f}")
        print(f"  Discrimination: {r['error_discrimination_ratio']:.1f}x")
        print(f"  U-Error corr: {r['uncertainty_error_correlation']:.4f}")

    return results


if __name__ == "__main__":
    run_calibration_experiment()
