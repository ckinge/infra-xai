"""
GraphSAGE 编码器：BILGR 的核心 backbone

基于 GraphSAGE (Hamilton et al., NeurIPS 2017) 的归纳式图编码器
支持 3 种聚合器: mean, max, lstm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional, List, Tuple


class SAGEEncoder(nn.Module):
    """
    多层 GraphSAGE 编码器

    参数与 BILGR 论文一致:
    - K=3 层
    - 每层后跟 BatchNorm + ReLU
    - 输出为节点嵌入向量
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggregator: str = 'mean',
        normalize: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第一层: in_channels -> hidden_channels
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, aggr=aggregator, normalize=normalize)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 中间层: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggregator, normalize=normalize)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 最后一层: hidden_channels -> out_channels
        self.convs.append(
            SAGEConv(hidden_channels, out_channels, aggr=aggregator, normalize=normalize)
        )
        self.bns.append(nn.BatchNorm1d(out_channels))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        dropout: Optional[float] = None
    ) -> torch.Tensor:
        """
        前向传播，dropout 参数支持 MC Dropout 时的动态设置

        Args:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
            dropout: 覆盖默认 dropout（MC Dropout 时设为 None 以使用 train mode）

        Returns:
            节点嵌入 [N, out_channels]
        """
        drop = dropout if dropout is not None else self.dropout_rate

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=drop, training=self.training)
        return x

    def get_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """获取节点嵌入（最后一层输出前的特征）"""
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns[:-1])):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 最后一层卷积但不带BN和激活，得到最终嵌入
        x = self.convs[-1](x, edge_index)
        return x
