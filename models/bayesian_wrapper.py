"""
贝叶斯包装器：MC Dropout + MAP图估计
处理 BILGR 中的两类不确定性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from torch_geometric.data import Data
from scipy.spatial.distance import cdist


class MAPGraphEstimator:
    """
    基于平滑信号优化的 MAP 图估计
    用于处理 aleatoric uncertainty（图结构噪声）

    原理: 假设节点嵌入在图上是平滑变化的，优化一个图结构
          使其既与观测图接近，又能解释节点嵌入的平滑性
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        """
        Args:
            alpha: 图结构保真度权重（高→更接近观测图）
            beta: 平滑正则化权重（高→更强调嵌入平滑性）
        """
        self.alpha = alpha
        self.beta = beta

    def estimate(
        self,
        obs_adj: torch.Tensor,
        embeddings: torch.Tensor,
        n_iter: int = 50,
    ) -> torch.Tensor:
        """
        MAP 图估计

        Args:
            obs_adj: 观测邻接矩阵 [N, N]
            embeddings: 节点嵌入 [N, D]
            n_iter: 优化迭代次数

        Returns:
            估计的加权邻接矩阵 [N, N]
        """
        N = embeddings.shape[0]
        device = embeddings.device

        # 初始化: 观测图 + 少量噪声
        W = obs_adj.clone()
        # 计算嵌入距离矩阵
        Z = torch.tensor(cdist(embeddings.cpu().detach().numpy(),
                                embeddings.cpu().detach().numpy()),
                        device=device)

        # 简单的梯度下降来逼近MAP解
        lr = 0.01
        for _ in range(n_iter):
            # 图结构保真度损失
            fidelity_loss = torch.norm(W - obs_adj, p='fro')

            # 平滑正则化: tr(E^T L E) = Σ_{i,j} W_{ij} ||e_i - e_j||^2
            smooth_loss = 0.0
            for i in range(N):
                for j in range(N):
                    if W[i, j] > 1e-6:
                        smooth_loss += W[i, j] * Z[i, j] ** 2

            total_loss = self.alpha * fidelity_loss + self.beta * smooth_loss

            # 梯度更新
            grad = 2 * self.alpha * (W - obs_adj)
            for i in range(N):
                for j in range(N):
                    grad[i, j] += self.beta * Z[i, j] ** 2

            W = W - lr * grad
            # 投影: 非负 + 对称
            W = torch.clamp(W, min=0.0)
            W = (W + W.T) / 2

        # 归一化
        W = W / (W.sum() + 1e-10) * obs_adj.sum()
        return W


class MCDropoutWrapper:
    """
    Monte Carlo Dropout 包装器
    用于处理 epistemic uncertainty（模型权重不确定性）

    在推理时执行多次前向传播（dropout开启），
    得到预测分布，计算均值和不确定性
    """

    def __init__(self, model: nn.Module, n_samples: int = 30):
        """
        Args:
            model: 包含 dropout 层的模型
            n_samples: MC 采样次数
        """
        self.model = model
        self.n_samples = n_samples

    def predict(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC Dropout 预测

        Returns:
            mean: 平均预测 logits [N, C]
            variance: 预测方差（不确定性）[N, C]
            samples: 所有采样 [S, N, C]
        """
        # 强制启用 dropout（训练模式但不算梯度）
        was_training = self.model.training
        self.model.train()

        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self.model(x, edge_index)
                samples.append(logits.unsqueeze(0))

        # 恢复原状态
        self.model.train(was_training)

        samples = torch.cat(samples, dim=0)  # [S, N, C]
        mean = samples.mean(dim=0)           # [N, C]
        variance = samples.var(dim=0)        # [N, C]

        return mean, variance, samples

    def predict_with_uncertainty(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测 + 总体不确定性

        Returns:
            predictions: 预测类别 [N]
            uncertainty: 总不确定性分数 [N]（0-1，越高越不确定）
        """
        mean, variance, samples = self.predict(x, edge_index)

        # 预测类别
        predictions = mean.argmax(dim=-1)

        # 不确定性: 预测熵 + 方差
        probs = mean.softmax(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        total_var = variance.sum(dim=-1)

        # 归一化到 [0,1]
        uncertainty = entropy + total_var
        uncertainty = (uncertainty - uncertainty.min()) / (
            uncertainty.max() - uncertainty.min() + 1e-10
        )

        return predictions, uncertainty
