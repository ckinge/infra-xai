"""
反事实推理模块 — 核心创新 #2

回答: "如果加固节点X，级联故障风险降低多少？"
为基础设施规划者提供可操作的韧性提升建议
"""
import torch
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data


class CounterfactualAnalyzer:
    """
    反事实推理器

    核心问题:
    - "What if we strengthen node X?" → 级联故障范围变化
    - "What if we remove dependency Y?" → 韧性指标变化
    - "Which K interventions give the maximum resilience gain?"
    """

    def __init__(
        self,
        model,
        cascade_simulator,
        top_k_budget: int = 5,
    ):
        """
        Args:
            model: BILGR 模型
            cascade_simulator: 级联故障模拟器
            top_k_budget: 可加固的节点数量预算
        """
        self.model = model
        self.cascade = cascade_simulator
        self.budget = top_k_budget

    def simulate_intervention(
        self,
        data: Data,
        strengthened_node: int,
        strength_factor: float = 2.0,
        initial_failures: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        模拟单个节点的加固干预效果

        使用相同的初始故障集合，确保公平对比
        """
        # 固定初始故障（如果未指定，生成一次并在两次模拟中复用）
        if initial_failures is None:
            n_init = max(1, int(data.num_nodes * 0.05))
            np.random.seed(42)
            initial_failures = list(np.random.choice(data.num_nodes, n_init, replace=False))

        # 1. 原始级联故障模拟
        original_result = self.cascade.simulate(
            data, initial_failures=initial_failures,
        )
        original_size = len(original_result['failed_nodes'])

        # 2. 加固干预: 修改节点容量
        modified_data = self._strengthen_node(data, strengthened_node, strength_factor)

        # 3. 反事实模拟（相同初始故障）
        counterfactual_result = self.cascade.simulate(
            modified_data, initial_failures=initial_failures,
        )
        cf_size = len(counterfactual_result['failed_nodes'])

        # 4. 计算效果
        risk_reduction = (
            (original_size - cf_size) / max(original_size, 1)
        )
        resilience_gain = (
            counterfactual_result.get('resilience', 0.0) -
            original_result.get('resilience', 0.0)
        )

        return {
            'original_failure_size': original_size,
            'counterfactual_failure_size': cf_size,
            'risk_reduction': risk_reduction,
            'resilience_gain': resilience_gain,
            'strengthened_node': strengthened_node,
        }

    def _strengthen_node(
        self, data: Data, node_idx: int, factor: float = 2.0
    ) -> Data:
        """加固节点：将其容量加倍"""
        import copy
        import networkx as nx

        modified = copy.deepcopy(data)

        # 计算节点的实际初始负载（与 cascase simulator 一致）
        G = modified.nx_graph if hasattr(modified, 'nx_graph') else None
        if G is None:
            # rebuild from edge_index
            G = nx.Graph()
            edges = modified.edge_index.t().tolist()
            G.add_edges_from(edges)

        neighbors = list(G.neighbors(node_idx))
        deg = len(neighbors)
        neighbor_deg = sum(len(list(G.neighbors(n))) for n in neighbors) if neighbors else 0
        initial_load = float(deg + 0.1 * neighbor_deg)

        # 初始化所有节点的容量（如果尚未设置）
        if not hasattr(modified, 'node_capacity') or modified.node_capacity is None:
            modified.node_capacity = torch.ones(modified.num_nodes) * (-1.0)

        # 设置加固节点的容量 = factor × 初始负载
        modified.node_capacity[node_idx] = initial_load * factor * 1.5  # factor * capacity_factor

        return modified

    def rank_interventions(
        self,
        data: Data,
        candidate_nodes: List[int],
        strength_factor: float = 2.0,
    ) -> List[Dict[str, float]]:
        """
        对所有候选节点的加固效果排序

        Returns:
            按 risk_reduction 降序排列的干预效果列表
        """
        results = []
        for node in candidate_nodes:
            result = self.simulate_intervention(data, node, strength_factor)
            results.append(result)

        results.sort(key=lambda x: x['risk_reduction'], reverse=True)
        return results

    def find_optimal_intervention_set(
        self,
        data: Data,
        candidate_nodes: List[int],
    ) -> List[int]:
        """
        贪婪搜索最优加固组合（在预算约束下）

        每次选择在当前状态下效果最好的节点
        """
        remaining_budget = self.budget
        selected = []
        current_data = copy.deepcopy(data)

        while remaining_budget > 0 and len(candidate_nodes) > len(selected):
            # 评估每个剩余候选节点
            best_node = None
            best_reduction = -float('inf')

            for node in candidate_nodes:
                if node in selected:
                    continue
                result = self.simulate_intervention(
                    current_data, node,
                )
                if result['risk_reduction'] > best_reduction:
                    best_reduction = result['risk_reduction']
                    best_node = node

            if best_node is not None:
                selected.append(best_node)
                # 更新当前状态（加固已选节点）
                current_data = self._strengthen_node(current_data, best_node)
                remaining_budget -= 1
            else:
                break

        return selected


class InterventionStrategy:
    """
    干预策略对比器

    比较不同策略的韧性提升效果:
    1. 基于GNN解释的干预
    2. 基于拓扑指标的干预
    3. 随机干预 (baseline)
    """

    def __init__(self, counterfactual: CounterfactualAnalyzer):
        self.cf = counterfactual

    def compare_strategies(
        self,
        data: Data,
        explainability_scores: torch.Tensor,
        topo_scores: torch.Tensor,
        n_candidates: int = 20,
    ) -> Dict[str, List[float]]:
        """
        对比三种干预策略

        Returns:
            {
                'explainability_guided': [各预算下的风险降低],
                'topology_guided': [...],
                'random': [...],
            }
        """
        N = data.num_nodes

        # 各策略的候选节点
        _, explain_candidates = torch.topk(explainability_scores, n_candidates)
        _, topo_candidates = torch.topk(topo_scores, n_candidates)
        random_candidates = torch.randperm(N)[:n_candidates]

        results = {}
        for strategy, candidates in [
            ('explainability_guided', explain_candidates.tolist()),
            ('topology_guided', topo_candidates.tolist()),
            ('random', random_candidates.tolist()),
        ]:
            optimal = self.cf.find_optimal_intervention_set(
                data, candidates,
            )
            # 评估最优组合效果
            cumulative_risk_reduction = []
            for k in range(1, len(optimal) + 1):
                subset = optimal[:k]
                total_reduction = 0.0
                for node in subset:
                    r = self.cf.simulate_intervention(data, node)
                    total_reduction += r['risk_reduction']
                cumulative_risk_reduction.append(total_reduction / k)

            results[strategy] = cumulative_risk_reduction

        return results
