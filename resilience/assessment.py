"""
多维度韧性评估框架
整合 GNN 预测 + 可解释性 + 韧性指标
"""
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.data import Data
from dataclasses import dataclass, field

from .metrics import ResilienceMetrics
from .simulation import CascadeSimulator


@dataclass
class ResilienceReport:
    """完整的韧性评估报告"""
    network_id: str = ""
    num_nodes: int = 0
    num_edges: int = 0

    # 韧性指标
    resilience_metrics: Dict[str, float] = field(default_factory=dict)

    # 关键节点
    critical_nodes: List[int] = field(default_factory=list)
    critical_scores: List[float] = field(default_factory=list)
    uncertain_nodes: List[int] = field(default_factory=list)

    # 级联故障
    cascade_vulnerability: float = 0.0
    expected_failure_size: int = 0

    # 可解释性
    top_explanation_features: Dict[str, Any] = field(default_factory=dict)

    # 建议
    recommended_interventions: List[int] = field(default_factory=list)
    expected_risk_reduction: float = 0.0


class ResilienceAssessor:
    """
    多维度韧性评估器

    综合使用:
    1. GNN 预测 → 识别关键节点
    2. 可解释性 → 理解脆弱性来源
    3. 韧性指标 → 量化系统韧性
    4. 级联模拟 → 验证预测
    5. 反事实推理 → 提出改进建议
    """

    def __init__(
        self,
        model=None,
        explainer_factory=None,
        cascade_sim: Optional[CascadeSimulator] = None,
    ):
        self.model = model
        self.explainer_factory = explainer_factory
        self.cascade_sim = cascade_sim or CascadeSimulator()

    def assess(
        self,
        data: Data,
        G: nx.Graph,
        network_id: str = "unknown",
    ) -> ResilienceReport:
        """
        执行完整的多维度韧性评估

        Args:
            data: PyG Data
            G: networkx 图
            network_id: 网络标识符

        Returns:
            ResilienceReport
        """
        report = ResilienceReport()
        report.network_id = network_id
        report.num_nodes = data.num_nodes
        report.num_edges = data.edge_index.shape[1]

        # 1. 计算韧性指标
        report.resilience_metrics = ResilienceMetrics.compute_all(G)

        # 2. 使用模型识别关键节点
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(data.x, data.edge_index)
                critical_scores = logits.softmax(dim=-1)[:, 0]  # Class 0 = 高关键性

            _, indices = torch.sort(critical_scores, descending=True)
            report.critical_nodes = indices[:20].tolist()
            report.critical_scores = critical_scores[indices[:20]].tolist()

        # 3. 级联模拟
        cascade_result = self.cascade_sim.simulate(
            data, failure_ratio=0.05,
        )
        report.cascade_vulnerability = 1.0 - cascade_result['resilience']
        report.expected_failure_size = len(cascade_result['failed_nodes'])

        return report

    def compare_networks(
        self, reports: List[ResilienceReport]
    ) -> Dict[str, List[float]]:
        """
        跨网络韧性对比
        """
        return {
            'network_ids': [r.network_id for r in reports],
            'resilience_scores': [r.resilience_metrics.get('robustness_auc', 0)
                                 for r in reports],
            'cascade_vulnerability': [r.cascade_vulnerability for r in reports],
            'num_critical_nodes': [len(r.critical_nodes) for r in reports],
        }
