"""
韧性评估指标体系
"""
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data


class ResilienceMetrics:
    """多维度韧性评估指标"""

    @staticmethod
    def effective_graph_resistance(G: nx.Graph) -> float:
        """有效图电阻: R_g = 2/(N-1) * Σ(1/λ_i)"""
        N = G.number_of_nodes()
        if N <= 1:
            return 0.0
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(L)
        non_zero = eigenvalues[eigenvalues > 1e-10]
        if len(non_zero) == 0:
            return float('inf')
        return (2.0 / (N - 1)) * np.sum(1.0 / non_zero)

    @staticmethod
    def algebraic_connectivity(G: nx.Graph) -> float:
        """代数连通度 (Fiedler值): 第二小特征值"""
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(L)
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    @staticmethod
    def natural_connectivity(G: nx.Graph) -> float:
        """自然连通度: ln((1/N) * Σ exp(λ_i))"""
        A = nx.adjacency_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(A)
        N = G.number_of_nodes()
        return float(np.log(np.sum(np.exp(eigenvalues)) / N))

    @staticmethod
    def largest_component_ratio(G: nx.Graph) -> float:
        """最大连通分量占比"""
        if G.number_of_nodes() == 0:
            return 0.0
        largest = len(max(nx.connected_components(G), key=len))
        return largest / G.number_of_nodes()

    @staticmethod
    def average_path_length(G: nx.Graph) -> float:
        """平均最短路径（效率指标）"""
        try:
            return float(nx.average_shortest_path_length(G))
        except Exception:
            return float('inf')

    @staticmethod
    def robustness_curve(
        G: nx.Graph, attack_type: str = 'degree', n_removals: int = 20
    ) -> np.ndarray:
        """
        鲁棒性曲线: 逐步移除节点后最大连通分量大小的变化

        Returns:
            每一步后的 LCC 比例数组
        """
        G_copy = G.copy()
        lcc_sizes = [ResilienceMetrics.largest_component_ratio(G_copy)]

        for _ in range(min(n_removals, G_copy.number_of_nodes())):
            if G_copy.number_of_nodes() == 0:
                break

            # 选择攻击目标
            if attack_type == 'degree':
                target = max(G_copy.degree(), key=lambda x: x[1])[0]
            elif attack_type == 'betweenness':
                bc = nx.betweenness_centrality(G_copy)
                target = max(bc, key=bc.get)
            elif attack_type == 'random':
                target = np.random.choice(list(G_copy.nodes()))
            else:
                target = np.random.choice(list(G_copy.nodes()))

            G_copy.remove_node(target)
            lcc_sizes.append(ResilienceMetrics.largest_component_ratio(G_copy))

        return np.array(lcc_sizes)

    @staticmethod
    def resilience_triangle(
        G_before: nx.Graph, G_after: nx.Graph
    ) -> Dict[str, float]:
        """
        韧性三角: 评估攻击前后的韧性损失

        Returns:
            {
                'robustness_loss': 鲁棒性损失,
                'recovery_potential': 恢复潜力,
                'adaptation_capacity': 适应能力,
            }
        """
        # 鲁棒性损失
        R_before = ResilienceMetrics.algebraic_connectivity(G_before)
        R_after = ResilienceMetrics.algebraic_connectivity(G_after)
        robustness_loss = (R_before - R_after) / max(R_before, 1e-10)

        # 最大连通分量损失
        lcc_before = ResilienceMetrics.largest_component_ratio(G_before)
        lcc_after = ResilienceMetrics.largest_component_ratio(G_after)
        structure_loss = lcc_before - lcc_after
        recovery_potential = 1.0 - structure_loss

        # 适应能力（剩余网络效率）
        if G_after.number_of_nodes() > 0:
            adaptation = 1.0 / (
                ResilienceMetrics.average_path_length(G_after) + 1.0
            )
        else:
            adaptation = 0.0

        return {
            'robustness_loss': robustness_loss,
            'recovery_potential': recovery_potential,
            'adaptation_capacity': adaptation,
        }

    @staticmethod
    def compute_all(
        G: nx.Graph, attack_type: str = 'degree', n_removals: int = 20
    ) -> Dict[str, float]:
        """计算所有韧性指标"""
        return {
            'effective_graph_resistance': ResilienceMetrics.effective_graph_resistance(G),
            'algebraic_connectivity': ResilienceMetrics.algebraic_connectivity(G),
            'natural_connectivity': ResilienceMetrics.natural_connectivity(G),
            'largest_component_ratio': ResilienceMetrics.largest_component_ratio(G),
            'average_path_length': ResilienceMetrics.average_path_length(G),
            'robustness_auc': float(np.trapz(
                ResilienceMetrics.robustness_curve(G, attack_type, n_removals)
            ) / (n_removals + 1)),
        }
