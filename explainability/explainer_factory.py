"""
可解释性模块工厂
统一接口，支持多种可解释性方法
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer


class ExplainabilityFactory:
    """
    可解释器工厂类
    统一管理 GNNExplainer, PGExplainer, IntegratedGradients 等方法
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        config: Optional[Any] = None,
    ):
        self.model = model
        self.device = device
        self.cfg = config
        self.model.to(device)

    def get_model_config(self, task_level: str = 'node') -> Dict:
        """构建 model_config 字典"""
        return {
            'mode': 'multiclass_classification',
            'task_level': task_level,
            'return_type': 'log_probs',
        }

    def build_gnnexplainer(
        self, task_level: str = 'node'
    ) -> Explainer:
        """
        构建 GNNExplainer (实例级，无需训练)
        """
        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(
                epochs=self.cfg.gnnexplainer_epochs if self.cfg else 200,
                lr=self.cfg.gnnexplainer_lr if self.cfg else 0.01,
            ),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=self.get_model_config(task_level),
        )
        return explainer

    def build_pgexplainer(
        self, task_level: str = 'node'
    ) -> Explainer:
        """
        构建 PGExplainer (参数化，需要训练后归纳)
        """
        explainer = Explainer(
            model=self.model,
            algorithm=PGExplainer(
                epochs=self.cfg.pgexplainer_epochs if self.cfg else 30,
                lr=self.cfg.pgexplainer_lr if self.cfg else 0.003,
            ),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=self.get_model_config(task_level),
            threshold_config=dict(threshold_type='topk', value=10),
        )
        return explainer

    def build_all_explainers(
        self, task_level: str = 'node'
    ) -> Dict[str, Explainer]:
        """构建所有可解释器"""
        return {
            'GNNExplainer': self.build_gnnexplainer(task_level),
            'PGExplainer': self.build_pgexplainer(task_level),
        }


def evaluate_explanation_fidelity(
    model: nn.Module,
    data: Data,
    explanation: Any,
    index: int,
    top_k: int = 20,
) -> float:
    """
    Evaluate explanation fidelity via predicted-class probability drop.
    This is the standard XAI definition:
      Fidelity = Pr(predicted class | original graph) - Pr(predicted class | masked graph)
    Positive value = removing important edges reduces confidence in the predicted class.
    Signed fidelity is reported (not clipped to 0), so negative values indicate cases
    where edge removal paradoxically *increases* confidence.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        orig_out = model(data.x.to(device), data.edge_index.to(device))
        orig_prob = orig_out[index].softmax(dim=-1)
        pred_class = orig_prob.argmax().item()
        orig_conf = orig_prob[pred_class].item()

    edge_mask = None
    if hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None:
        edge_mask = explanation.edge_mask
    elif hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
        nm = explanation.node_mask
        if nm.dim() > 1:
            nm = nm.sum(dim=-1)
        edge_mask = torch.zeros(data.edge_index.shape[1], device=nm.device)
        for e in range(data.edge_index.shape[1]):
            src, dst = data.edge_index[0, e], data.edge_index[1, e]
            edge_mask[e] = (nm[src].abs() + nm[dst].abs()) / 2.0

    if edge_mask is None or edge_mask.abs().sum() < 1e-10:
        deg = torch.zeros(data.num_nodes, device=device)
        for e in range(data.edge_index.shape[1]):
            deg[data.edge_index[0, e]] += 1
        edge_mask = deg[data.edge_index[0]] + deg[data.edge_index[1]]

    _, top_indices = torch.topk(edge_mask.abs(), min(top_k, len(edge_mask)))
    keep_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)
    keep_mask[top_indices] = False
    masked_edge_index = data.edge_index[:, keep_mask].to(device)

    with torch.no_grad():
        masked_out = model(data.x.to(device), masked_edge_index)
        masked_prob = masked_out[index].softmax(dim=-1)
        masked_conf = masked_prob[pred_class].item()

    return orig_conf - masked_conf  # signed fidelity, no clipping


def fidelity_summary(fidelities: list) -> dict:
    """Compute aggregate fidelity statistics from per-node values."""
    import numpy as np
    arr = np.array(fidelities)
    return {
        'mean_signed': float(np.mean(arr)),
        'positive_rate': float((arr > 0).mean()),
        'median': float(np.median(arr)),
        'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        'std': float(np.std(arr)),
    }


def evaluate_explanation_sparsity(explanation: Any, top_k: int = 30) -> float:
    """
    评估解释的稀疏度

    Sparsity = 1 - (Top-K重要边 / 总边数)
    越高越好：用更少的边解释预测（实际论文中只看Top-K）
    """
    if hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None:
        mask = explanation.edge_mask
        n_edges = len(mask)
        if n_edges == 0:
            return 0.0
        # 稀疏度 = 1 - 重要边占比（这里用top_k作为"重要"的定义）
        sparsity = 1.0 - min(top_k / n_edges, 1.0)
        return sparsity
    return 0.0


def evaluate_explanation_stability(
    explanations: List[Any],
) -> float:
    """
    评估解释的稳定性

    对同一节点多次解释的边掩码相关性
    越高越好: 解释应一致
    """
    if len(explanations) < 2:
        return 1.0

    masks = []
    for exp in explanations:
        if hasattr(exp, 'edge_mask') and exp.edge_mask is not None:
            masks.append(exp.edge_mask.detach())

    if len(masks) < 2:
        return 1.0

    # 计算平均成对相关系数
    correlations = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            corr = torch.corrcoef(torch.stack([
                masks[i].flatten(), masks[j].flatten()
            ]))[0, 1].item()
            correlations.append(corr)

    return float(torch.tensor(correlations).mean().item())
