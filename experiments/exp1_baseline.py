"""
实验1: 可解释性方法基准比较
比较 GNNExplainer, Degree-based, Random 三种解释方法的性能
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from data.dataset_loader import DataPipeline
from models.bilgr import BILGR
from explainability.explainer_factory import (
    ExplainabilityFactory,
    evaluate_explanation_fidelity,
    evaluate_explanation_sparsity,
)
from tqdm import tqdm
import json


class BaselineExplainer:
    """简单的基线解释器: 基于度数或随机"""

    def __init__(self, data, method='degree'):
        self.data = data
        self.method = method

        if method == 'degree':
            # 边重要性 = 两端节点度之和
            deg = torch.zeros(data.num_nodes)
            for e in range(data.edge_index.shape[1]):
                src, dst = data.edge_index[0, e], data.edge_index[1, e]
                deg[src] += 1
                deg[dst] += 1
            self.edge_mask = deg[data.edge_index[0]] + deg[data.edge_index[1]]
            self.edge_mask = self.edge_mask / self.edge_mask.max()  # 归一化
        elif method == 'random':
            self.edge_mask = torch.rand(data.edge_index.shape[1])

    def __call__(self, x, edge_index, index=None):
        """模拟 PyG explainer 的调用接口"""
        from types import SimpleNamespace
        return SimpleNamespace(
            edge_mask=self.edge_mask,
            node_mask=None,
        )


def run_experiment_1():
    """可解释性方法基准比较实验"""
    print("=" * 60)
    print("Experiment 1: Explainability Method Benchmark")
    print(f"Device: {cfg.experiment.device}")
    print("=" * 60)

    device = cfg.experiment.device
    results = {}

    # 为不同网络类型运行
    for net_type in ['scale_free', 'small_world', 'random']:
        print(f"\n--- Network: {net_type} ---")

        # 1. 准备数据
        pipeline = DataPipeline(cfg)
        data = pipeline.run(
            "synthetic",
            n_nodes=500,
            network_type=net_type,
        ).to(device)

        # 检查类别分布
        for c in range(3):
            count = (data.y == c).sum().item()
            print(f"  Class {c}: {count} nodes ({count/data.num_nodes*100:.1f}%)")

        # 2. 训练 BILGR 模型
        model = BILGR(
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.model.out_channels,
            num_layers=cfg.model.num_layers,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.experiment.lr,
            weight_decay=cfg.experiment.weight_decay,
        )

        print(f"  Training ({cfg.experiment.epochs} epochs)...")
        model.train()
        for epoch in range(cfg.experiment.epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = torch.nn.functional.cross_entropy(
                out[data.train_mask], data.y[data.train_mask]
            )
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    out_eval = model(data.x, data.edge_index)
                    pred = out_eval.argmax(dim=-1)
                    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, val_acc={val_acc:.4f}")
                model.train()

        # 最终评估
        model.eval()
        with torch.no_grad():
            out_final = model(data.x, data.edge_index)
            pred_final = out_final.argmax(dim=-1)
            train_acc = (pred_final[data.train_mask] == data.y[data.train_mask]).float().mean()
            val_acc = (pred_final[data.val_mask] == data.y[data.val_mask]).float().mean()
            test_acc = (pred_final[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f"  Final: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

        # 3. 构建解释器
        factory = ExplainabilityFactory(model, device, cfg.explain)
        gnnexplainer = factory.build_gnnexplainer()
        degree_explainer = BaselineExplainer(data, method='degree')
        random_explainer = BaselineExplainer(data, method='random')

        explainers = {
            'GNNExplainer': gnnexplainer,
            'Degree': degree_explainer,
            'Random': random_explainer,
        }

        # 4. 评估每种解释方法
        test_nodes = torch.where(data.test_mask)[0][:30]  # 取30个测试节点
        method_results = {}

        for method_name, explainer in explainers.items():
            print(f"  Evaluating {method_name} ({len(test_nodes)} test nodes)...")

            fidelities = []
            sparsities = []

            # 非 GNNExplainer 不需要 tqdm（太快了）
            node_iter = tqdm(test_nodes.tolist(), desc=f"    {method_name}") if method_name == 'GNNExplainer' else test_nodes.tolist()

            for node_idx in node_iter:
                try:
                    explanation = explainer(
                        data.x, data.edge_index,
                        index=node_idx,
                    )

                    fid = evaluate_explanation_fidelity(
                        model, data, explanation, node_idx,
                    )
                    spar = evaluate_explanation_sparsity(explanation)

                    fidelities.append(fid)
                    sparsities.append(spar)
                except Exception as e:
                    if method_name == 'GNNExplainer':
                        print(f"      Warning: node {node_idx} failed: {e}")

            method_results[method_name] = {
                'fidelity_mean': float(np.mean(fidelities)) if fidelities else 0,
                'fidelity_std': float(np.std(fidelities)) if fidelities else 0,
                'sparsity_mean': float(np.mean(sparsities)) if sparsities else 0,
                'sparsity_std': float(np.std(sparsities)) if sparsities else 0,
                'num_successful': len(fidelities),
            }

        results[net_type] = method_results

    # 5. 保存结果
    os.makedirs(cfg.experiment.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.experiment.output_dir, "exp1_benchmark.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 6. 打印总结
    print("\n" + "=" * 60)
    print("Experiment 1 Summary")
    print("=" * 60)
    for net_type, methods in results.items():
        print(f"\n{net_type}:")
        for method, metrics in methods.items():
            print(f"  {method}: Fidelity={metrics['fidelity_mean']:.4f}±{metrics['fidelity_std']:.4f}, "
                  f"Sparsity={metrics['sparsity_mean']:.4f}±{metrics['sparsity_std']:.4f} "
                  f"({metrics['num_successful']} nodes)")

        # 计算 GNNExplainer 相对度数的提升
        if 'GNNExplainer' in methods and 'Degree' in methods:
            gn_fid = methods['GNNExplainer']['fidelity_mean']
            deg_fid = methods['Degree']['fidelity_mean']
            if deg_fid > 0:
                improvement = (gn_fid - deg_fid) / deg_fid * 100
                print(f"  → GNNExplainer improvement over Degree: {improvement:+.1f}%")

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_experiment_1()
