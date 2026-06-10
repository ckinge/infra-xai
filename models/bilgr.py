"""
BILGR: Bayesian Inductive Learner for Graph Robustness
完整模型（复现自 Munikoti et al., IEEE SMC 2021）

架构:
  GraphSAGE Encoder (K=3) → MLP Classifier → 3-class Criticality
  + MC Dropout (epistemic uncertainty)
  + MAP Graph Estimation (aleatoric uncertainty)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch_geometric.data import Data

from .graphsage_encoder import SAGEEncoder
from .bayesian_wrapper import MAPGraphEstimator, MCDropoutWrapper


class BILGRClassifier(nn.Module):
    """BILGR 分类器: GraphSAGE → MLP → 3-class"""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = SAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x, edge_index)
        return self.classifier(embeddings)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """获取分类前的节点嵌入"""
        return self.encoder(x, edge_index)


class BILGR(nn.Module):
    """
    Bayesian Inductive Learner for Graph Robustness

    完整的贝叶斯 GNN 框架，用于识别不确定复杂网络中的关键节点
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
        mc_samples: int = 30,
        graph_samples: int = 10,
        map_alpha: float = 1.0,
        map_beta: float = 0.1,
    ):
        super().__init__()
        self.classifier = BILGRClassifier(
            in_channels, hidden_channels, out_channels,
            num_layers, num_classes, dropout,
        )
        self.mc_wrapper = MCDropoutWrapper(self.classifier, n_samples=mc_samples)
        self.map_estimator = MAPGraphEstimator(alpha=map_alpha, beta=map_beta)
        self.graph_samples = graph_samples
        self.num_classes = num_classes

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """标准前向传播（训练时使用）"""
        return self.classifier(x, edge_index)

    def bayesian_predict(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        obs_adj: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        贝叶斯预测：结合 MAP 图估计和 MC Dropout

        Args:
            x: 节点特征
            edge_index: 观测图边索引
            obs_adj: 观测图邻接矩阵（可选，用于 MAP 估计）

        Returns:
            {
                'predictions': 预测类别 [N],
                'uncertainty': 不确定性分数 [N],
                'logits_mean': 平均 logits [N, C],
                'logits_var': logits 方差 [N, C],
                'embeddings': 节点嵌入 [N, D],
                'estimated_graphs': MAP 估计图列表 (如果提供了 obs_adj),
            }
        """
        results = {}

        # 获取节点嵌入
        embeddings = self.classifier.get_embeddings(x, edge_index)
        results['embeddings'] = embeddings

        # 如果有观测图，进行 MAP 估计
        if obs_adj is not None:
            est_graphs = []
            for _ in range(self.graph_samples):
                est_W = self.map_estimator.estimate(
                    obs_adj, embeddings,
                )
                est_graphs.append(est_W)
            results['estimated_graphs'] = est_graphs

        # MC Dropout 预测
        mean, variance, samples = self.mc_wrapper.predict(x, edge_index)
        predictions = mean.argmax(dim=-1)

        # 计算不确定性
        probs = mean.softmax(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        total_var = variance.sum(dim=-1)
        uncertainty = entropy + total_var
        uncertainty = (uncertainty - uncertainty.min()) / (
            uncertainty.max() - uncertainty.min() + 1e-10
        )

        results.update({
            'predictions': predictions,
            'uncertainty': uncertainty,
            'logits_mean': mean,
            'logits_var': variance,
            'prediction_distribution': samples,
        })

        return results

    def get_critical_nodes(
        self, data: Data, top_k: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        识别 Top-K 关键节点（带不确定性）

        Returns:
            critical_indices: Top-K 节点索引
            critical_scores: 对应分数
        """
        adj = torch.zeros((data.num_nodes, data.num_nodes), device=data.x.device)
        adj[data.edge_index[0], data.edge_index[1]] = 1.0

        results = self.bayesian_predict(data.x, data.edge_index, obs_adj=adj)

        # 关键性分数: softmax 的 Class 0 (高关键性) 概率
        critical_scores = results['logits_mean'].softmax(dim=-1)[:, 0]

        # 排序
        _, indices = torch.sort(critical_scores, descending=True)
        top_indices = indices[:top_k]
        top_scores = critical_scores[top_indices]

        return top_indices, top_scores
