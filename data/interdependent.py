"""
相互依赖关键基础设施网络构建器
基于 Buldyrev et al. (2010) 的相互依赖网络模型
"""
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Tuple, List, Dict, Optional
from .dataset_loader import InfrastructureDataLoader


class InterdependentNetworkBuilder:
    """
    相互依赖网络构建器

    生成两个耦合网络 A 和 B，具有:
    - 内部边 (intra-edges): 各网络内部的拓扑连接
    - 依赖边 (inter-dependency edges): 跨网络的依赖关系
      A 中的节点依赖于 B 中的节点（或双向）
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def build_coupled_networks(
        self,
        n_nodes_a: int = 500,
        n_nodes_b: int = 500,
        topo_a: str = "scale_free",
        topo_b: str = "scale_free",
        n_dependencies: int = 150,
        dependency_direction: str = "bidirectional",
    ) -> Tuple[nx.Graph, nx.Graph, List[Tuple[int, int]]]:
        """
        构建耦合网络对

        Args:
            n_nodes_a, n_nodes_b: 两个网络的节点数
            topo_a, topo_b: 拓扑类型
            n_dependencies: 跨网络依赖边数量
            dependency_direction: 'bidirectional' | 'a_to_b' | 'b_to_a'

        Returns:
            G_a, G_b: 两个网络
            dependencies: [(node_in_a, node_in_b), ...] 依赖关系列表
        """
        loader = InfrastructureDataLoader()

        G_a = loader.generate_synthetic_network(
            n_nodes=n_nodes_a, network_type=topo_a
        )
        G_b = loader.generate_synthetic_network(
            n_nodes=n_nodes_b, network_type=topo_b
        )

        # 生成随机依赖边
        nodes_a = list(G_a.nodes())
        nodes_b = list(G_b.nodes())

        # 优先连接度较高的节点（更符合现实）
        deg_a = np.array([G_a.degree(n) for n in nodes_a], dtype=float)
        deg_b = np.array([G_b.degree(n) for n in nodes_b], dtype=float)

        prob_a = deg_a / deg_a.sum()
        prob_b = deg_b / deg_b.sum()

        dependencies = []
        used_a = set()
        used_b = set()

        for _ in range(n_dependencies):
            # 加权随机采样
            a = np.random.choice(nodes_a, p=prob_a)
            b = np.random.choice(nodes_b, p=prob_b)
            dependencies.append((a, b))
            used_a.add(a)
            used_b.add(b)

        return G_a, G_b, dependencies

    def _compute_interdependent_labels(
        self,
        G_a: nx.Graph,
        G_b: nx.Graph,
        dependencies: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        使用相互依赖级联模拟生成关键性标签。

        对每个节点（A和B共计 N_a+N_b 个），从耦合网络中移除该节点，
        然后在相同初始故障集下运行相互依赖级联模拟。
        标签 = 移除该节点后剩余耦合系统的级联规模增幅，按百分位数分三档。
        """
        N_a = G_a.number_of_nodes()
        N_b = G_b.number_of_nodes()
        N_total = N_a + N_b

        np.random.seed(42)
        n_init = max(1, int(N_a * 0.05))
        init_fails = [int(x) for x in np.random.choice(N_a, n_init, replace=False)]

        def make_sim_data(
            Ga: nx.Graph, Gb: nx.Graph, deps: List[Tuple[int, int]]
        ) -> Data:
            sim_data = Data(num_nodes=N_total)
            sim_data.nx_graph_a = Ga
            sim_data.nx_graph_b = Gb
            sim_data.dependencies = deps
            sim_data.n_a = N_a
            sim_data.n_b = N_b
            return sim_data

        def filter_dependencies(
            deps: List[Tuple[int, int]], Ga: nx.Graph, Gb: nx.Graph
        ) -> List[Tuple[int, int]]:
            active_a = set(Ga.nodes())
            active_b = set(Gb.nodes())
            return [(a, b) for a, b in deps if a in active_a and b in active_b]

        def remove_node(
            node_idx: int,
        ) -> Tuple[nx.Graph, nx.Graph, List[Tuple[int, int]], List[int]]:
            Ga = G_a.copy()
            Gb = G_b.copy()
            if node_idx < N_a:
                if node_idx in Ga:
                    Ga.remove_node(node_idx)
            else:
                b_node = node_idx - N_a
                if b_node in Gb:
                    Gb.remove_node(b_node)

            filtered_deps = filter_dependencies(dependencies, Ga, Gb)
            filtered_init = [n for n in init_fails if n in Ga.nodes()]
            return Ga, Gb, filtered_deps, filtered_init

        simulator = InterdependentCascadeSimulator(
            capacity_factor=1.5, max_steps=30
        )

        # 基线级联
        baseline_deps = filter_dependencies(dependencies, G_a, G_b)
        baseline_data = make_sim_data(G_a, G_b, baseline_deps)
        baseline_total = simulator.simulate(
            baseline_data, initial_failures_a=init_fails
        )['total_failed']

        # 逐节点移除
        criticality = np.zeros(N_total)
        for i in range(N_total):
            Ga_i, Gb_i, deps_i, init_i = remove_node(i)
            data_i = make_sim_data(Ga_i, Gb_i, deps_i)
            total = simulator.simulate(
                data_i, initial_failures_a=init_i
            )['total_failed']
            score = max(0.0, (total - baseline_total) / max(N_total, 1))
            criticality[i] = score

        # 百分位数分三档
        p33 = np.percentile(criticality, 33.33)
        p67 = np.percentile(criticality, 66.67)
        discrete = np.zeros(N_total, dtype=np.int64)
        if p67 > p33:
            discrete[criticality < p33] = 2
            discrete[(criticality >= p33) & (criticality < p67)] = 1
            discrete[criticality >= p67] = 0
        else:
            for i in range(N_total):
                discrete[i] = i % 3
        return discrete

    def build_coupled_pyg_data(
        self,
        G_a: nx.Graph,
        G_b: nx.Graph,
        dependencies: List[Tuple[int, int]],
    ) -> Data:
        """
        构建包含两个网络 + 依赖关系的 PyG Data 对象

        采用"超级节点"方案: 将所有节点放在同一个图中
        节点 0..N_a-1 属于网络A
        节点 N_a..N_a+N_b-1 属于网络B
        三种边:
          1. A内部边 (intra-a)
          2. B内部边 (intra-b)
          3. 依赖边 (dependency edges, A↔B)

        添加 edge_type 属性区分三种边
        """
        loader = InfrastructureDataLoader()
        N_a = G_a.number_of_nodes()
        N_b = G_b.number_of_nodes()
        N_total = N_a + N_b

        # 偏移 B 的节点索引
        edges = []

        def add_edge(u: int, v: int, edge_type_id: int):
            """Add a physical/dependency relation as bidirectional message edges."""
            edges.append([u, v, edge_type_id])
            if u != v:
                edges.append([v, u, edge_type_id])

        # A内部边
        for u, v in G_a.edges():
            add_edge(u, v, 0)  # type 0 = intra-A

        # B内部边（偏移 N_a）
        for u, v in G_b.edges():
            add_edge(u + N_a, v + N_a, 1)  # type 1 = intra-B

        # 依赖边 (A↔B)
        for a, b in dependencies:
            add_edge(a, b + N_a, 2)  # type 2 = dependency

        if edges:
            edge_index = torch.tensor(
                [[e[0], e[1]] for e in edges], dtype=torch.long
            ).t().contiguous()
            edge_type = torch.tensor([e[2] for e in edges], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0,), dtype=torch.long)

        # 节点特征
        feat_a = loader.compute_node_features(G_a)
        feat_b = loader.compute_node_features(G_b)
        x = torch.tensor(
            np.vstack([feat_a, feat_b]), dtype=torch.float
        )

        # 节点标签 —— 使用相互依赖级联模拟标签
        # 标签反映的是"移除该节点后，在耦合系统中引发的总级联规模增幅"
        labels = self._compute_interdependent_labels(
            G_a, G_b, dependencies
        )
        y = torch.tensor(labels, dtype=torch.long)

        # 网络归属标签
        network_label = torch.zeros(N_total, dtype=torch.long)
        network_label[N_a:] = 1  # B网络节点标记为1

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=N_total,
            edge_type=edge_type,
            network_label=network_label,
            n_a=N_a,
            n_b=N_b,
            n_deps=len(dependencies),
        )

        # 保存原始图用于级联模拟
        data.nx_graph_a = G_a
        data.nx_graph_b = G_b
        data.dependencies = dependencies

        return data


class InterdependentCascadeSimulator:
    """
    相互依赖网络级联故障模拟器

    扩展 Motter-Lai 模型到相互依赖网络:
    1. 网络A中的初始故障
    2. A中故障 → A中负载重分布 → A中更多故障
    3. A中故障 → 依赖节点在B中故障（跨网络传播）
    4. B中故障 → B中负载重分布 → B中更多故障
    5. B中故障 → 依赖节点在A中故障（回流）
    6. 重复直到稳定
    """

    def __init__(
        self,
        capacity_factor: float = 1.5,
        max_steps: int = 30,
    ):
        self.capacity_factor = capacity_factor
        self.max_steps = max_steps

    def simulate(
        self,
        data: Data,
        initial_failures_a: Optional[List[int]] = None,
        failure_ratio: float = 0.05,
    ) -> Dict:
        """
        模拟相互依赖网络中的级联故障

        Args:
            data: 耦合网络数据
            initial_failures_a: A网络中的初始故障节点
            failure_ratio: 初始故障比例
        """
        G_a = data.nx_graph_a
        G_b = data.nx_graph_b
        raw_deps: List[Tuple[int, int]] = data.dependencies
        N_a = data.n_a
        N_b = data.n_b
        active_nodes_a = set(G_a.nodes())
        active_nodes_b = set(G_b.nodes())
        deps = [
            (a, b) for a, b in raw_deps
            if a in active_nodes_a and b in active_nodes_b
        ]

        # 构建依赖映射
        a_to_b = {}  # A节点 → 依赖的B节点列表
        b_to_a = {}  # B节点 → 依赖的A节点列表
        for a, b in deps:
            a_to_b.setdefault(a, []).append(b)
            b_to_a.setdefault(b, []).append(a)

        # 初始负载
        loads_a = self._compute_loads(G_a)
        loads_b = self._compute_loads(G_b)

        # 检查节点级容量覆盖（用于反事实实验）
        has_override = hasattr(data, 'node_capacity') and data.node_capacity is not None
        caps_a = {}
        caps_b = {}
        for n in G_a.nodes():
            if has_override and data.node_capacity[n].item() > 0:
                caps_a[n] = data.node_capacity[n].item()
            else:
                caps_a[n] = loads_a[n] * self.capacity_factor
        for n in G_b.nodes():
            idx = n + N_a  # B网络节点在总数组中的偏移
            if has_override and data.node_capacity[idx].item() > 0:
                caps_b[n] = data.node_capacity[idx].item()
            else:
                caps_b[n] = loads_b[n] * self.capacity_factor

        # 初始故障
        if initial_failures_a is None:
            n_init = max(1, int(N_a * failure_ratio))
            candidate_a = sorted(active_nodes_a)
            if candidate_a:
                n_init = min(n_init, len(candidate_a))
                initial_failures_a = list(
                    np.random.choice(candidate_a, n_init, replace=False)
                )
            else:
                initial_failures_a = []
        else:
            initial_failures_a = [
                n for n in initial_failures_a if n in active_nodes_a
            ]

        failed_a = set(initial_failures_a)
        failed_b = set()
        cascade_steps = [{'a': list(failed_a), 'b': []}]

        for step in range(self.max_steps):
            new_failed_a = set()
            new_failed_b = set()

            # A中的负载重分布
            active_a = set(G_a.nodes()) - failed_a
            if len(active_a) > 1:
                active_sub = G_a.subgraph(active_a)
                new_loads_a = self._compute_loads(active_sub)
                # 故障邻居的额外负载
                for n in active_a:
                    failed_neighbors = sum(1 for nb in G_a.neighbors(n) if nb in failed_a)
                    new_loads_a[n] = new_loads_a.get(n, 0) * (1.0 + 0.5 * failed_neighbors)

                for n, load in new_loads_a.items():
                    if load > caps_a.get(n, 1.0):
                        new_failed_a.add(n)

            # B中的负载重分布
            active_b = set(G_b.nodes()) - failed_b
            if len(active_b) > 1:
                active_sub = G_b.subgraph(active_b)
                new_loads_b = self._compute_loads(active_sub)
                for n in active_b:
                    failed_neighbors = sum(1 for nb in G_b.neighbors(n) if nb in failed_b)
                    new_loads_b[n] = new_loads_b.get(n, 0) * (1.0 + 0.5 * failed_neighbors)

                for n, load in new_loads_b.items():
                    if load > caps_b.get(n, 1.0):
                        new_failed_b.add(n)

            # 跨网络传播: A中新增故障 → B中依赖节点故障
            for a_node in new_failed_a:
                if a_node in a_to_b:
                    for b_node in a_to_b[a_node]:
                        if b_node not in failed_b:
                            new_failed_b.add(b_node)

            # 跨网络传播: B中新增故障 → A中依赖节点故障
            for b_node in new_failed_b:
                if b_node in b_to_a:
                    for a_node in b_to_a[b_node]:
                        if a_node not in failed_a:
                            new_failed_a.add(a_node)

            if not new_failed_a and not new_failed_b:
                break

            failed_a.update(new_failed_a)
            failed_b.update(new_failed_b)
            cascade_steps.append({
                'a': list(new_failed_a),
                'b': list(new_failed_b),
            })

        total_failed = len(failed_a) + len(failed_b)
        total_nodes = N_a + N_b

        return {
            'failed_a': failed_a,
            'failed_b': failed_b,
            'total_failed': total_failed,
            'cascade_steps': cascade_steps,
            'resilience': 1.0 - total_failed / total_nodes,
            'total_steps': len(cascade_steps),
            'cross_propagation': len(failed_b) > 0,
        }

    def _compute_loads(self, G: nx.Graph) -> Dict[int, float]:
        loads = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            deg = len(neighbors)
            nd = sum(len(list(G.neighbors(n))) for n in neighbors) if neighbors else 0
            loads[node] = float(deg + 0.1 * nd)
        return loads
