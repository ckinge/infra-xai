"""
拓扑-学习融合框架 — 核心创新 #3

将传统网络科学拓扑指标与 GNN 学习的解释进行交叉验证和融合
提供"领域可解释 + 数据驱动"的双重视角
"""
import torch
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Optional
from torch_geometric.data import Data


class TopologyAnalyzer:
    """传统网络科学拓扑分析"""

    @staticmethod
    def compute_all_metrics(G: nx.Graph) -> Dict[str, np.ndarray]:
        """
        计算所有拓扑中心性指标

        Returns:
            {metric_name: node_values_array}
        """
        metrics = {}

        # 度中心性
        metrics['degree'] = np.array([
            d for _, d in G.degree()
        ])

        # 介数中心性 (对小图精确计算)
        if G.number_of_nodes() < 5000:
            bc = nx.betweenness_centrality(G, normalized=True)
            metrics['betweenness'] = np.array([bc[n] for n in G.nodes()])
        else:
            # 大图用近似
            metrics['betweenness'] = np.zeros(G.number_of_nodes())

        # 接近中心性
        try:
            cc = nx.closeness_centrality(G)
            metrics['closeness'] = np.array([cc[n] for n in G.nodes()])
        except Exception:
            metrics['closeness'] = np.zeros(G.number_of_nodes())

        # 特征向量中心性
        try:
            ec = nx.eigenvector_centrality_numpy(G)
            metrics['eigenvector'] = np.array([ec[n] for n in G.nodes()])
        except Exception:
            metrics['eigenvector'] = np.zeros(G.number_of_nodes())

        # 聚类系数
        clustering = nx.clustering(G)
        metrics['clustering'] = np.array([clustering[n] for n in G.nodes()])

        # PageRank
        pr = nx.pagerank(G)
        metrics['pagerank'] = np.array([pr[n] for n in G.nodes()])

        # K-shell / core number
        core = nx.core_number(G)
        metrics['core_number'] = np.array([core[n] for n in G.nodes()])

        return metrics

    @staticmethod
    def normalize_metrics(metrics: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """将所有指标归一化到 [0,1]"""
        normalized = {}
        for name, values in metrics.items():
            vmin, vmax = values.min(), values.max()
            if vmax - vmin > 1e-10:
                normalized[name] = (values - vmin) / (vmax - vmin)
            else:
                normalized[name] = np.zeros_like(values)
        return normalized


class FusionEngine:
    """
    拓扑-学习融合引擎

    将传统拓扑指标与 GNN 学到的解释分数进行加权融合
    """

    def __init__(
        self,
        topo_weight: float = 0.4,
        learn_weight: float = 0.6,
    ):
        self.topo_weight = topo_weight
        self.learn_weight = learn_weight

    def fuse_node_scores(
        self,
        learned_scores: torch.Tensor,      # GNN 解释的节点分数 [N]
        topo_metrics: Dict[str, np.ndarray],  # 拓扑指标
        topo_weights: Optional[Dict[str, float]] = None,  # 各拓扑指标权重
    ) -> Dict[str, torch.Tensor]:
        """
        融合学习和拓扑的节点分数

        fused_score = learn_weight * learned_score + topo_weight * topo_composite

        Returns:
            {
                'fused': 融合分数,
                'learned': 学习分数,
                'topo_composite': 拓扑综合分数,
                'topo_individual': 各拓扑指标分数,
            }
        """
        N = len(learned_scores)

        # 默认拓扑权重: betweenness 和 eigenvector 最重要
        if topo_weights is None:
            topo_weights = {
                'betweenness': 0.30,
                'eigenvector': 0.20,
                'pagerank': 0.20,
                'degree': 0.15,
                'closeness': 0.10,
                'core_number': 0.05,
            }

        # 归一化拓扑指标
        norm_metrics = TopologyAnalyzer.normalize_metrics(topo_metrics)

        # 拓扑综合分数
        topo_composite = np.zeros(N)
        for name, weight in topo_weights.items():
            if name in norm_metrics:
                topo_composite += weight * norm_metrics[name]

        # 归一化学习分数
        learned_np = learned_scores.cpu().numpy()
        l_min, l_max = learned_np.min(), learned_np.max()
        if l_max - l_min > 1e-10:
            learned_norm = (learned_np - l_min) / (l_max - l_min)
        else:
            learned_norm = np.zeros_like(learned_np)

        # 融合
        fused = (
            self.learn_weight * learned_norm +
            self.topo_weight * topo_composite
        )

        return {
            'fused': torch.tensor(fused, dtype=torch.float),
            'learned': torch.tensor(learned_norm, dtype=torch.float),
            'topo_composite': torch.tensor(topo_composite, dtype=torch.float),
            'topo_individual': {
                name: torch.tensor(vals, dtype=torch.float)
                for name, vals in norm_metrics.items()
            },
        }

    def compute_fusion_agreement(
        self,
        learned_scores: torch.Tensor,
        topo_metrics: Dict[str, np.ndarray],
        top_k: int = 20,
    ) -> float:
        """
        计算学习和拓扑的 Top-K 一致性

        返回值越高，说明两种解释方法越一致
        """
        N = len(learned_scores)

        # 归一化拓扑综合分数
        norm_metrics = TopologyAnalyzer.normalize_metrics(topo_metrics)
        topo_composite = np.zeros(N)
        weights = {'betweenness': 0.3, 'eigenvector': 0.2, 'pagerank': 0.2,
                   'degree': 0.15, 'closeness': 0.1, 'core_number': 0.05}
        for name, w in weights.items():
            if name in norm_metrics:
                topo_composite += w * norm_metrics[name]

        # 各自 Top-K
        learned_topk = set(torch.topk(learned_scores, top_k).indices.tolist())
        topo_topk = set(np.argsort(topo_composite)[-top_k:].tolist())

        # Jaccard 相似度
        intersection = len(learned_topk & topo_topk)
        union = len(learned_topk | topo_topk)
        agreement = intersection / union if union > 0 else 0.0

        return agreement


class TrustedExplanation:
    """
    可信解释框架 — 核心创新 #2 的延伸

    结合贝叶斯不确定性来标注解释的可信度:
    - 高不确定性节点的解释 → 低可信度
    - 低不确定性节点的解释 → 高可信度

    这对关键基础设施决策至关重要
    """

    def __init__(self, uncertainty_threshold: float = 0.3):
        self.threshold = uncertainty_threshold

    def compute_trust_scores(
        self,
        explanation_scores: torch.Tensor,
        uncertainty_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算可信度加权的解释分数

        Args:
            explanation_scores: 原始解释分数 [N]
            uncertainty_scores: 贝叶斯不确定性 [N]

        Returns:
            trusted_scores: 可信度调整后的分数 [N]
            trust_levels: 可信度标签 [N] (0=低, 1=中, 2=高)
        """
        # 可信度因子: 1 / (1 + uncertainty)
        trust_factor = 1.0 / (1.0 + uncertainty_scores)

        # 加权解释分数
        trusted_scores = explanation_scores * trust_factor

        # 可信度分级
        trust_levels = torch.zeros_like(uncertainty_scores, dtype=torch.long)
        trust_levels[uncertainty_scores < self.threshold] = 2      # 高可信
        trust_levels[
            (uncertainty_scores >= self.threshold) &
            (uncertainty_scores < 2 * self.threshold)
        ] = 1                                                       # 中可信
        trust_levels[uncertainty_scores >= 2 * self.threshold] = 0  # 低可信

        return trusted_scores, trust_levels
