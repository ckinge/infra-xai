"""
全局配置文件
"""
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """BILGR 模型配置"""
    # GraphSAGE 编码器 (BILGR原论文参数, 经调参验证为最优)
    in_channels: int = 2          # 输入特征: 加权度 + 平均邻居度
    hidden_channels: int = 64     # 隐藏层维度
    out_channels: int = 64        # 嵌入维度
    num_layers: int = 3           # GraphSAGE 层数 (K=3, BILGR原论文)
    dropout: float = 0.3          # Dropout (MC Dropout 复用此值)

    # 分类器
    num_classes: int = 3          # 3类关键性: 高/中/低

    # 贝叶斯
    mc_samples: int = 30          # MC Dropout 采样次数
    graph_samples: int = 10       # MAP 图采样次数
    uncertainty_threshold: float = 0.3  # 高不确定性阈值


@dataclass
class ExplainabilityConfig:
    """可解释性模块配置"""
    # GNNExplainer
    gnnexplainer_epochs: int = 200
    gnnexplainer_lr: float = 0.01

    # PGExplainer
    pgexplainer_epochs: int = 30
    pgexplainer_lr: float = 0.003
    pgexplainer_hidden: int = 64

    # 多粒度
    top_k_nodes: int = 20         # 节点级 top-k
    top_k_edges: int = 30         # 边级 top-k
    subgraph_radius: int = 2      # 子图级 hop 数

    # 融合
    topo_weight: float = 0.4      # 拓扑先验权重
    learn_weight: float = 0.6     # 学习解释权重

    # 评估
    fidelity_threshold: float = 0.7
    sparsity_target: float = 0.3


@dataclass
class ResilienceConfig:
    """韧性评估配置"""
    initial_failure_ratio: float = 0.05   # 初始故障比例
    cascade_threshold: float = 0.5        # 级联传播阈值
    max_cascade_steps: int = 20           # 最大级联步数
    recovery_budget: int = 5              # 加固预算（反事实实验）
    capacity_decay: float = 0.8           # 每次过载后的容量衰减


@dataclass
class ExperimentConfig:
    """实验配置"""
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 训练
    epochs: int = 200
    lr: float = 0.001
    weight_decay: float = 5e-4
    batch_size: int = 64
    patience: int = 30             # 早停

    # 数据
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # 评估
    num_repeats: int = 5          # 重复实验次数

    # 输出
    output_dir: str = "./outputs"
    save_model: bool = True
    log_interval: int = 10


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    explain: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典，便于保存"""
        return {
            "model": self.model.__dict__,
            "explain": self.explain.__dict__,
            "resilience": self.resilience.__dict__,
            "experiment": self.experiment.__dict__,
        }


# 全局默认配置
cfg = Config()
