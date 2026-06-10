"""
可视化模块: 解释结果和评估指标的可视化
"""
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from typing import Dict, List, Optional, Any
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})


def plot_explanation_heatmap(
    data: Data,
    explanation: Any,
    node_idx: int,
    save_path: str,
    title: str = "Explanation Heatmap",
):
    """
    将解释结果可视化为网络热力图
    节点大小 = 关键性, 边透明度 = 解释重要性
    """
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 边宽度基于解释重要性
    if hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None:
        edge_weights = explanation.edge_mask.cpu().numpy()
        edge_weights = (edge_weights - edge_weights.min()) / (
            edge_weights.max() - edge_weights.min() + 1e-10
        )
    else:
        edge_weights = np.ones(data.edge_index.shape[1]) * 0.1

    # 画边
    edge_list = [(data.edge_index[0, i].item(), data.edge_index[1, i].item())
                 for i in range(data.edge_index.shape[1])]

    for (u, v), w in zip(edge_list, edge_weights):
        alpha = 0.05 + w * 0.95
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                'k-', alpha=alpha, linewidth=w * 3)

    # 画节点
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if n == node_idx:
            node_colors.append('red')
            node_sizes.append(200)
        else:
            node_colors.append('steelblue')
            node_sizes.append(30)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=node_sizes, alpha=0.8)

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    metric: str = 'fidelity_mean',
    title: str = "Method Comparison",
):
    """柱状图对比不同方法的指标"""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results.keys())
    values = []
    errors = []

    for method, metrics in results.items():
        if isinstance(metrics, dict):
            values.append(metrics.get(metric, 0))
            errors.append(metrics.get(metric.replace('mean', 'std'), 0))
        else:
            values.append(0)
            errors.append(0)

    x = np.arange(len(methods))
    bars = ax.bar(x, values, yerr=errors, capsize=5,
                  color=sns.color_palette("viridis", len(methods)))

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_cascade_curve(
    failure_curves: Dict[str, List[int]],
    save_path: str,
    title: str = "Cascading Failure Dynamics",
):
    """级联故障动态曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, curve in failure_curves.items():
        ax.plot(range(len(curve)), curve, 'o-', label=label, linewidth=2,
               markersize=6)

    ax.set_xlabel("Cascade Step")
    ax.set_ylabel("Cumulative Failed Nodes")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_counterfactual_comparison(
    strategy_results: Dict[str, List[float]],
    save_path: str,
    title: str = "Intervention Strategy Comparison",
):
    """反事实干预策略对比"""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {'explainability_guided': '#2ecc71', 'topology_guided': '#3498db',
              'random': '#e74c3c'}
    labels_cn = {'explainability_guided': '可解释性引导', 'topology_guided': '拓扑引导',
                  'random': '随机'}

    for strategy, values in strategy_results.items():
        if values:
            ax.plot(range(1, len(values) + 1), values, 'o-',
                   color=colors.get(strategy, 'gray'),
                   label=labels_cn.get(strategy, strategy),
                   linewidth=2, markersize=6)

    ax.set_xlabel("Intervention Budget (K)")
    ax.set_ylabel("Cumulative Risk Reduction")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_ablation_waterfall(
    ablation_results: Dict[str, Dict],
    save_path: str,
    baseline_key: str = 'gcnn_only',
):
    """消融实验瀑布图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline_key in ablation_results:
        baseline_fid = ablation_results[baseline_key]['fidelity_mean']
    else:
        baseline_fid = 0.0

    names = []
    improvements = []
    for name, res in ablation_results.items():
        if name != baseline_key:
            names.append(name)
            improvements.append(res['fidelity_mean'] - baseline_fid)

    # 瀑布图: 从 baseline 开始累积
    cumulative = [baseline_fid]
    labels_pos = [baseline_key]
    for name, imp in zip(names, improvements):
        cumulative.append(cumulative[-1] + imp)
        labels_pos.append(name)

    colors = ['steelblue'] + ['#2ecc71' if x > 0 else '#e74c3c'
                              for x in improvements]
    x = np.arange(len(labels_pos))

    ax.bar(x, [baseline_fid] + improvements, bottom=[0] +
           [cumulative[i] - max(0, improvements[i])
            for i in range(len(improvements))],
           color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_pos, rotation=30, ha='right')
    ax.set_ylabel("Fidelity Score")
    ax.set_title("Ablation Study: Module Contribution")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
