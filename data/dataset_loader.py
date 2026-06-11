"""
数据集加载器：支持多种基础设施网络数据源
"""
import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Tuple, List, Optional, Dict
import scipy.io as sio


class InfrastructureDataLoader:
    """
    多源关键基础设施网络数据加载器
    支持:
    - IEEE 多域基础设施网络 (GraphML)
    - Kostia-Zuev 电网拓扑 (.mat/.csv)
    - 合成网络 (scale-free, small-world, random)
    """

    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = data_dir
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)

    # ─── 公开数据集加载 ───────────────────────────────

    def load_ieee_multidomain(self, filepath: str) -> nx.DiGraph:
        """加载IEEE多域基础设施网络 (GraphML格式)"""
        G = nx.read_graphml(filepath)
        return G

    def load_power_grid_mat(self, filepath: str) -> nx.Graph:
        """加载Kostia-Zuev电网 (.mat格式)"""
        mat = sio.loadmat(filepath)
        # 提取邻接矩阵
        adj_key = [k for k in mat.keys() if not k.startswith('__')][0]
        adj = mat[adj_key]
        if isinstance(adj, np.ndarray) and adj.ndim == 2:
            G = nx.from_numpy_array(adj)
        else:
            # 尝试从 struct 中提取
            G = nx.from_scipy_sparse_array(adj) if hasattr(adj, 'toarray') else nx.Graph()
        return G

    def load_matpower(self, filepath: str) -> nx.Graph:
        """
        解析 MATPOWER case 文件 (.m 格式)，提取电网拓扑图

        MATPOWER 格式:
          mpc.bus = [bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
          mpc.branch = [fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax]

        Returns:
            networkx 图，节点带地理属性，边带电抗权重
        """
        import re

        with open(filepath, 'r') as f:
            content = f.read()

        # 提取 bus 数据矩阵
        bus_match = re.search(r'mpc\.bus\s*=\s*\[(.*?)\];', content, re.DOTALL)
        if not bus_match:
            raise ValueError("Cannot find mpc.bus in file")

        # 提取 branch 数据矩阵
        branch_match = re.search(r'mpc\.branch\s*=\s*\[(.*?)\];', content, re.DOTALL)
        if not branch_match:
            raise ValueError("Cannot find mpc.branch in file")

        def parse_matrix(text):
            """解析MATLAB矩阵，处理分号分隔的行"""
            rows = []
            # 按分号分割行
            raw_rows = text.split(';')
            for row in raw_rows:
                # 清理空白和注释
                row = row.strip()
                if not row or row.startswith('%'):
                    continue
                # 按空格/逗号分割数值
                # MATLAB 中空格和逗号都是分隔符
                row = row.replace(',', ' ')
                values = []
                for v in row.split():
                    try:
                        values.append(float(v))
                    except ValueError:
                        continue
                if values:
                    rows.append(values)
            return rows

        bus_data = parse_matrix(bus_match.group(1))
        branch_data = parse_matrix(branch_match.group(1))

        # 构建图：节点 = 母线，边 = 输电线路
        G = nx.Graph()

        # 添加节点（母线）
        for i, bus in enumerate(bus_data):
            G.add_node(i, bus_id=int(bus[0]) if len(bus) > 0 else i+1)

        # 添加边（线路）
        for branch in branch_data:
            if len(branch) < 2:
                continue
            fbus = int(branch[0]) - 1  # MATPOWER 从1开始编号
            tbus = int(branch[1]) - 1
            status = int(branch[10]) if len(branch) > 10 else 1

            if status <= 0:  # 停运线路跳过
                continue

            # 电抗作为边的物理权重
            reactance = abs(branch[3]) if len(branch) > 3 else 1.0
            if reactance < 1e-8:
                reactance = 1.0

            G.add_edge(fbus, tbus, weight=1.0 / reactance)

        return G

    def load_power_grid_edgelist(self, filepath: str) -> nx.Graph:
        """加载边列表格式的电网数据"""
        edges = np.loadtxt(filepath, delimiter=',')
        G = nx.Graph()
        G.add_edges_from(edges[:, :2].astype(int))
        return G

    # ─── 合成网络生成 ────────────────────────────────

    def generate_synthetic_network(
        self,
        n_nodes: int = 500,
        network_type: str = "scale_free",
        **kwargs
    ) -> nx.Graph:
        """
        生成合成网络用于受控实验

        Args:
            n_nodes: 节点数
            network_type: scale_free | small_world | random | clustered

        Returns:
            networkx 图
        """
        if network_type == "scale_free":
            m = kwargs.get("m", 3)
            G = nx.barabasi_albert_graph(n_nodes, m)

        elif network_type == "small_world":
            k = kwargs.get("k", 6)
            p = kwargs.get("p", 0.1)
            G = nx.watts_strogatz_graph(n_nodes, k, p)

        elif network_type == "random":
            p = kwargs.get("p", 0.05)
            G = nx.erdos_renyi_graph(n_nodes, p)

        elif network_type == "clustered":
            # 生成带有社区结构的网络
            sizes = kwargs.get("sizes", [n_nodes // 4] * 4)
            p_in = kwargs.get("p_in", 0.3)
            p_out = kwargs.get("p_out", 0.01)
            G = nx.random_partition_graph(sizes, p_in, p_out)

        else:
            raise ValueError(f"Unknown network type: {network_type}")

        return G

    # ─── 图特征工程 ─────────────────────────────────

    def compute_node_features(self, G: nx.Graph) -> np.ndarray:
        """
        计算节点特征矩阵
        特征1: 加权度（带自环）
        特征2: 平均邻居度
        与 BILGR 原论文保持一致
        """
        n = G.number_of_nodes()
        features = np.zeros((n, 2))

        # 特征1: 加权度
        for i, node in enumerate(G.nodes()):
            degree = G.degree(node, weight='weight') if nx.is_weighted(G) else G.degree(node)
            features[i, 0] = degree + 1  # +1 自环

        # 特征2: 平均邻居度
        for i, node in enumerate(G.nodes()):
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 0:
                neighbor_degrees = [
                    G.degree(n, weight='weight') if nx.is_weighted(G) else G.degree(n)
                    for n in neighbors
                ]
                features[i, 1] = np.mean(neighbor_degrees)
            else:
                features[i, 1] = 0.0

        return features

    def compute_criticality_labels(
        self, G: nx.Graph, n_classes: int = 3
    ) -> np.ndarray:
        """
        Compute structural criticality labels using weighted topology metrics.

        Uses a composite of betweenness centrality (50%), PageRank (30%),
        and degree centrality (20%) as a proxy for structural vulnerability.
        Labels are partitioned into balanced classes via percentile thresholds.

        Note on framing: These labels capture *topological criticality* — a
        proxy for, but not identical to, cascade-based resilience. For
        interdependent networks, cascade simulation labels are used instead
        (see InterdependentNetworkBuilder).
        """
        N = G.number_of_nodes()

        if N > 2000:
            k = min(N, 500)
            bc = nx.betweenness_centrality(G, k=k, normalized=True)
        else:
            bc = nx.betweenness_centrality(G, normalized=True)
        bc_array = np.array([bc[n] for n in G.nodes()])

        pr = nx.pagerank(G, alpha=0.85)
        pr_array = np.array([pr[n] for n in G.nodes()])

        dc_array = np.array([G.degree(n) for n in G.nodes()], dtype=np.float64)

        def normalize(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn > 1e-10:
                return (arr - mn) / (mx - mn)
            return np.zeros_like(arr)

        criticality = (0.50 * normalize(bc_array) +
                       0.30 * normalize(pr_array) +
                       0.20 * normalize(dc_array))

        p33 = np.percentile(criticality, 33.33)
        p67 = np.percentile(criticality, 66.67)

        discrete_labels = np.zeros(N, dtype=np.int64)
        if p67 > p33:
            discrete_labels[criticality < p33] = 2
            discrete_labels[(criticality >= p33) & (criticality < p67)] = 1
            discrete_labels[criticality >= p67] = 0
        else:
            for i in range(N):
                discrete_labels[i] = i % 3
        return discrete_labels

    # ─── PyG Data 转换 ───────────────────────────────

    def to_pyg_data(
        self, G: nx.Graph, features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Data:
        """将 networkx 图转换为 PyG Data 对象"""
        if features is None:
            features = self.compute_node_features(G)
        if labels is None:
            labels = self.compute_criticality_labels(G)

        # 构建无向边索引（显式添加反向边）
        # 基础设施网络（电网、通信网）的物理连接是双向的
        edges = []
        for u, v in G.edges():
            edges.append([u, v])
            if u != v:
                edges.append([v, u])  # 反向边，确保消息双向传播
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long),
            num_nodes=G.number_of_nodes(),
        )
        return data

    # ─── 数据分割 ─────────────────────────────────────

    def train_val_test_split(
        self, data: Data, train_r: float = 0.6, val_r: float = 0.2, test_r: float = 0.2
    ) -> Tuple[Data, Dict[str, torch.Tensor]]:
        """
        节点分类任务的数据分割
        返回增强的 Data 和掩码字典
        """
        n = data.num_nodes
        indices = np.random.permutation(n)

        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data


class DataPipeline:
    """
    完整数据处理流水线
    """

    def __init__(self, config=None):
        self.loader = InfrastructureDataLoader()
        self.cfg = config

    def run(self, source: str, **kwargs) -> Data:
        """一键运行数据处理流水线"""
        if source == "synthetic":
            G = self.loader.generate_synthetic_network(**kwargs)
        elif source == "power_grid_mat":
            G = self.loader.load_power_grid_mat(kwargs.get("filepath"))
        elif source == "power_grid_edgelist":
            G = self.loader.load_power_grid_edgelist(kwargs.get("filepath"))
        elif source == "ieee_multidomain":
            G = self.loader.load_ieee_multidomain(kwargs.get("filepath"))
        elif source == "matpower":
            G = self.loader.load_matpower(kwargs.get("filepath"))
        else:
            raise ValueError(f"Unknown source: {source}")

        features = self.loader.compute_node_features(G)
        labels = self.loader.compute_criticality_labels(G)
        data = self.loader.to_pyg_data(G, features, labels)
        data = self.loader.train_val_test_split(data)

        # 保存图结构信息用于后续拓扑分析
        data.nx_graph = G

        return data
