"""
多粒度可解释性模块 — 核心创新 #1

三层解释粒度:
  1. 节点级 — 哪些节点对级联传播最敏感
  2. 边级   — 哪些跨基础设施依赖驱动故障传播
  3. 子图级 — 哪些子网络最脆弱
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import Data
from torch_geometric.explain import Explainer
from torch_geometric.utils import k_hop_subgraph


class NodeLevelExplainer:
    """节点级解释器：识别关键节点及其解释特征"""

    def __init__(self, model: nn.Module, explainer: Explainer):
        self.model = model
        self.explainer = explainer

    def explain_nodes(
        self, data: Data, node_indices: List[int]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        对指定节点生成解释

        Returns:
            {node_idx: {'node_mask': ..., 'edge_mask': ..., 'feature_importance': ...}}
        """
        results = {}
        for idx in node_indices:
            explanation = self.explainer(data.x, data.edge_index, index=idx)
            results[idx] = {
                'node_mask': explanation.node_mask,
                'edge_mask': explanation.edge_mask,
                'prediction': self.model(data.x, data.edge_index)[idx].argmax().item(),
            }
        return results

    def rank_nodes_by_vulnerability(
        self, data: Data, top_k: int = 20
    ) -> torch.Tensor:
        """
        基于边缘重要性聚合的节点脆弱性排序

        节点脆弱性 = 连接到该节点的边的重要性之和
        """
        explanation = self.explainer(data.x, data.edge_index, index=0)
        if explanation.edge_mask is None:
            return torch.zeros(data.num_nodes)

        edge_mask = explanation.edge_mask
        node_score = torch.zeros(data.num_nodes)

        # 聚合连接到每个节点的边重要性
        # 无向基础设施网络已包含双向边，两端权重相等
        for e in range(data.edge_index.shape[1]):
            src = data.edge_index[0, e].item()
            dst = data.edge_index[1, e].item()
            score = edge_mask[e].item()
            node_score[src] += score
            node_score[dst] += score

        _, indices = torch.sort(node_score, descending=True)
        return indices[:top_k]


class EdgeLevelExplainer:
    """边级解释器：识别关键依赖关系和传播路径"""

    def __init__(self, model: nn.Module, explainer: Explainer):
        self.model = model
        self.explainer = explainer

    def explain_edges(
        self, data: Data, node_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        对指定节点的预测，返回边级重要性

        Returns:
            edge_mask: 边重要性 [E]
            edge_scores: 边重要性排序 [top_k_E]
            top_edges: Top-K 边的索引
        """
        explanation = self.explainer(data.x, data.edge_index, index=node_index)
        edge_mask = explanation.edge_mask

        if edge_mask is not None:
            _, top_indices = torch.sort(edge_mask, descending=True)
            return edge_mask, edge_mask[top_indices], top_indices.tolist()
        return torch.zeros(data.edge_index.shape[1]), torch.zeros(10), []

    def find_critical_paths(
        self, data: Data, source_nodes: List[int], top_k_per_source: int = 5
    ) -> Dict[int, List[int]]:
        """
        找出从关键源节点出发的最重要传播路径
        """
        critical_paths = {}
        for src in source_nodes:
            _, _, top_edges = self.explain_edges(data, src)
            critical_paths[src] = top_edges[:top_k_per_source]
        return critical_paths


class SubgraphExplainer:
    """子图级解释器：识别最脆弱的子网络"""

    def __init__(self, model: nn.Module, explainer: Explainer, radius: int = 2):
        self.model = model
        self.explainer = explainer
        self.radius = radius

    def _k_hop_subgraph_nx(self, G, center_node: int, radius: int):
        """
        使用 NetworkX 提取 k-hop 子图（比 PyG 的 k_hop_subgraph 更稳定）
        返回 PyG 格式的 (subset, sub_edge_index, center_idx)
        """
        import networkx as nx

        # 用 nx 提取 k-hop 邻域
        neighbors = nx.single_source_shortest_path_length(G, center_node, cutoff=radius)
        subset = sorted(neighbors.keys())
        node_to_idx = {n: i for i, n in enumerate(subset)}

        # 提取子图边
        sub_edges = []
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx:
                sub_edges.append([node_to_idx[u], node_to_idx[v]])
                if u != v:
                    sub_edges.append([node_to_idx[v], node_to_idx[u]])

        if len(sub_edges) == 0:
            sub_edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            sub_edge_index = torch.tensor(sub_edges, dtype=torch.long).t().contiguous()

        center_idx = node_to_idx[center_node]
        return subset, sub_edge_index, center_idx

    def extract_explanatory_subgraph(
        self, data: Data, center_node: int
    ) -> Tuple[Data, torch.Tensor, List[int]]:
        """
        提取中心节点周围的解释性子图

        Returns:
            subgraph: k-hop 子图 Data
            subgraph_mask: 子图内的边重要性
            subgraph_nodes: 子图节点列表
        """
        # 使用 NetworkX 提取 k-hop 子图（更稳定）
        G = data.nx_graph if hasattr(data, 'nx_graph') else None
        if G is not None:
            subset, sub_edge_index, center_idx = self._k_hop_subgraph_nx(
                G, center_node, self.radius
            )
        else:
            # Fallback: 使用 PyG 的 k_hop_subgraph
            from torch_geometric.utils import k_hop_subgraph
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                center_node, self.radius, data.edge_index,
                relabel_nodes=True,
            )
            center_idx = mapping.item()

        sub_data = Data(
            x=data.x[torch.tensor(subset, dtype=torch.long)],
            edge_index=sub_edge_index,
            num_nodes=len(subset),
        )
        sub_data.original_nodes = subset if isinstance(subset, list) else subset.tolist()

        # 在子图上重新解释
        explanation = self.explainer(
            sub_data.x, sub_data.edge_index, index=center_idx
        )

        return sub_data, explanation.edge_mask, sub_data.original_nodes

    def rank_subgraphs_by_vulnerability(
        self, data: Data, seed_nodes: List[int]
    ) -> List[Tuple[int, float]]:
        """
        对围绕种子节点的子图按脆弱性排序

        脆弱性 = 子图内边重要性均值 / 子图大小
        """
        scores = []
        for node in seed_nodes:
            _, edge_mask, nodes = self.extract_explanatory_subgraph(data, node)
            if edge_mask is not None and len(nodes) > 0:
                vulnerability = edge_mask.mean().item() / max(len(nodes), 1)
                scores.append((node, vulnerability))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class MultiGranularExplainer:
    """
    多粒度可解释性 — 统一接口

    整合节点级、边级、子图级解释，生成完整解释报告
    """

    def __init__(
        self, model: nn.Module, explainer: Explainer,
        config: Optional[Any] = None,
    ):
        self.model = model
        self.explainer = explainer
        self.cfg = config
        self.node_explainer = NodeLevelExplainer(model, explainer)
        self.edge_explainer = EdgeLevelExplainer(model, explainer)
        self.subgraph_explainer = SubgraphExplainer(model, explainer)

    def generate_full_report(
        self, data: Data,
        target_node: int,
        top_k_nodes: int = 20,
        top_k_edges: int = 30,
        subgraph_radius: int = 2,
    ) -> Dict[str, Any]:
        """
        对目标节点生成完整的多粒度解释报告

        这是论文中核心的输出结构
        """
        report = {}

        # 1. 节点级解释
        node_explanations = self.node_explainer.explain_nodes(
            data, [target_node]
        )
        vulnerable_nodes = self.node_explainer.rank_nodes_by_vulnerability(
            data, top_k=top_k_nodes
        )
        report['node_level'] = {
            'target_explanation': node_explanations,
            'vulnerable_nodes': vulnerable_nodes.tolist(),
        }

        # 2. 边级解释
        edge_mask, edge_scores, top_edges = self.edge_explainer.explain_edges(
            data, target_node
        )
        critical_paths = self.edge_explainer.find_critical_paths(
            data, [target_node], top_k_per_source=5
        )
        report['edge_level'] = {
            'edge_mask': edge_mask,
            'top_edges': top_edges[:top_k_edges],
            'critical_paths': critical_paths,
        }

        # 3. 子图级解释
        sub_data, sub_mask, sub_nodes = self.subgraph_explainer.extract_explanatory_subgraph(
            data, target_node
        )
        report['subgraph_level'] = {
            'subgraph_data': sub_data,
            'subgraph_mask': sub_mask,
            'subgraph_nodes': sub_nodes,
        }

        return report
