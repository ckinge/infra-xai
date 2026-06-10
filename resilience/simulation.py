"""
级联故障模拟器
模拟节点故障在基础设施网络中的传播过程
"""
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from torch_geometric.data import Data


class CascadeSimulator:
    """
    级联故障模拟器

    基于 Motter-Lai 模型的扩展版本:
    - 初始随机故障
    - 负载重分布
    - 节点过载 → 新故障
    - 迭代直到稳定
    """

    def __init__(
        self,
        capacity_factor: float = 1.5,
        capacity_decay: float = 0.8,
        max_steps: int = 20,
        overload_threshold: float = 1.0,
    ):
        """
        Args:
            capacity_factor: 容量 = 初始负载 × factor
            capacity_decay: 每次过载后容量衰减因子
            max_steps: 最大级联步数
            overload_threshold: 过载阈值（负载/容量）
        """
        self.capacity_factor = capacity_factor
        self.capacity_decay = capacity_decay
        self.max_steps = max_steps
        self.overload_threshold = overload_threshold

    def simulate(
        self,
        data: Data,
        initial_failures: Optional[List[int]] = None,
        failure_ratio: float = 0.05,
    ) -> Dict:
        """
        模拟级联故障

        Args:
            data: 图数据
            initial_failures: 指定的初始故障节点
            failure_ratio: 初始故障比例（当未指定 failures 时）

        Returns:
            {
                'failed_nodes': 最终故障节点集合,
                'cascade_steps': 每步新增故障节点,
                'failure_curve': 每步累计故障数,
                'resilience': 韧性分数 (1 - 最终故障比例),
            }
        """
        N = data.num_nodes
        G = self._build_nx_graph(data)

        # 如果未指定初始故障，随机选择
        if initial_failures is None:
            n_init = max(1, int(N * failure_ratio))
            initial_failures = list(np.random.choice(N, n_init, replace=False))

        # 计算节点初始负载和容量
        initial_loads = self._compute_initial_loads(G)

        # 检查是否有节点级容量覆盖（用于反事实实验）
        if hasattr(data, 'node_capacity') and data.node_capacity is not None:
            capacities = {}
            for node in G.nodes():
                node_cap = data.node_capacity[node].item()
                if node_cap > 0:  # 正值表示反事实容量覆盖
                    capacities[node] = node_cap
                else:
                    capacities[node] = initial_loads[node] * self.capacity_factor
        else:
            capacities = {
                node: load * self.capacity_factor
                for node, load in initial_loads.items()
            }

        # 级联过程
        all_failed = set(initial_failures)
        active = set(G.nodes()) - all_failed
        cascade_steps = [initial_failures.copy()]
        failure_curve = [len(initial_failures)]

        for step in range(self.max_steps):
            new_failures = set()

            # 重新计算活动子图上的负载
            if len(active) > 0:
                active_subgraph = G.subgraph(active)
                current_loads = self._compute_loads_on_subgraph(
                    G, active_subgraph, active, all_failed
                )
            else:
                break

            # 检查过载
            for node in active:
                load = current_loads.get(node, 0.0)
                cap = capacities.get(node, 1.0)
                if load / max(cap, 1e-10) > self.overload_threshold:
                    new_failures.add(node)
                    # 容量衰减
                    capacities[node] *= self.capacity_decay

            if not new_failures:
                break

            all_failed.update(new_failures)
            active -= new_failures
            cascade_steps.append(list(new_failures))
            failure_curve.append(len(all_failed))

        # 韧性分数
        resilience = 1.0 - len(all_failed) / N

        return {
            'failed_nodes': all_failed,
            'cascade_steps': cascade_steps,
            'failure_curve': failure_curve,
            'resilience': resilience,
            'total_steps': len(cascade_steps),
        }

    def _build_nx_graph(self, data: Data) -> nx.Graph:
        """PyG Data → networkx"""
        G = nx.Graph()
        edges = data.edge_index.t().tolist()
        G.add_edges_from(edges)
        return G

    def _compute_initial_loads(self, G: nx.Graph) -> Dict[int, float]:
        """
        基于加权度的初始负载 (O(E), 比介数中心性快100倍)
        degree是介数中心性的有效近似，适用于大规模网络
        """
        loads = {}
        for node in G.nodes():
            # 加权度：节点度 + 邻居度之和的加权
            neighbors = list(G.neighbors(node))
            deg = len(neighbors)
            neighbor_deg = sum(len(list(G.neighbors(n))) for n in neighbors) if neighbors else 0
            # 使用度中心性 + 邻居度作为负载代理
            loads[node] = float(deg + 0.1 * neighbor_deg)
        return loads

    def _compute_loads_on_subgraph(
        self,
        full_G: nx.Graph,
        sub_G: nx.Graph,
        active_nodes: Set[int],
        failed_nodes: Set[int],
    ) -> Dict[int, float]:
        """
        在活动子图上计算负载（使用加权度，快速近似）
        故障会引发流量重分布：邻居节点的负载增加
        """
        loads = {}
        for node in active_nodes:
            neighbors = list(sub_G.neighbors(node))
            deg = len(neighbors)
            # 邻居度加权
            neighbor_deg = sum(len(list(sub_G.neighbors(n))) for n in neighbors) if neighbors else 0
            base_load = float(deg + 0.1 * neighbor_deg)

            # 故障节点导致的额外负载：故障邻居数越多，负载越重
            failed_neighbors = sum(1 for n in full_G.neighbors(node) if n in failed_nodes)
            loads[node] = base_load * (1.0 + 0.5 * failed_neighbors)

        return loads
