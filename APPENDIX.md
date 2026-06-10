# 代码使用指南与改进方向

## 环境准备

```bash
cd infra-xai
pip install -r requirements.txt
```

## 与BILGR原始代码的关系

本项目的BILGR模型是saimunikoti/GraphNeuralNetwork-Resilience-ComplexNetworks的轻量复现版本。

**原始仓库结构** (基于分析):
```
saimunikoti/GraphNeuralNetwork-Resilience-ComplexNetworks/
├── src/              # 核心GNN代码 (GraphSAGE + MLP分类器)
├── src_bgnn/         # 贝叶斯GNN (先前的实现版本)
├── src_bilgr/        # BILGR (MC Dropout + MAP图估计)
├── src_pyth/         # Python工具
├── data/             # 数据集
├── models/           # 预训练模型
└── stellargraph/     # StellarGraph库集成
```

**我们复现的核心组件**:
- `models/graphsage_encoder.py` — GraphSAGE编码器 (K=3层，对应原BILGR)
- `models/bayesian_wrapper.py` — MC Dropout + MAP图估计
- `models/bilgr.py` — 完整BILGR模型

**我们新增的创新模块** (在原始代码中完全不存在):
- `explainability/` — 全部新增
- `resilience/` — 全部新增
- `experiments/` — 全部新增

## 核心文件说明

### 1. 模型训练

```python
from config import cfg
from data.dataset_loader import DataPipeline
from models.bilgr import BILGR

# 准备数据
pipeline = DataPipeline(cfg)
data = pipeline.run("synthetic", n_nodes=500, network_type="scale_free")

# 训练模型
model = BILGR(
    in_channels=2,         # 加权度 + 平均邻居度
    hidden_channels=64,
    out_channels=64,
    num_layers=3,
    dropout=0.3,
    mc_samples=30,         # MC Dropout采样次数
)
# ... 正常PyTorch训练循环
```

### 2. 可解释性分析

```python
from explainability.explainer_factory import ExplainabilityFactory
from explainability.multi_granular import MultiGranularExplainer

# 构建解释器
factory = ExplainabilityFactory(model, device, cfg.explain)
gnnexplainer = factory.build_gnnexplainer()

# 多粒度解释
multi = MultiGranularExplainer(model, gnnexplainer)
report = multi.generate_full_report(data, target_node=42)

# 获取各粒度解释
node_vuln = report['node_level']['vulnerable_nodes']    # Top-20脆弱节点
edge_crit = report['edge_level']['top_edges']            # Top-30关键边
sub_vuln = report['subgraph_level']['subgraph_nodes']    # 脆弱子图
```

### 3. 反事实推理

```python
from explainability.counterfactual import CounterfactualAnalyzer
from resilience.simulation import CascadeSimulator

cascade = CascadeSimulator(max_steps=20)
cf = CounterfactualAnalyzer(model, cascade, top_k_budget=5)

# 单个节点加固效果
result = cf.simulate_intervention(data, strengthened_node=10)
print(f"Risk reduction: {result['risk_reduction']:.2%}")

# 对比三种策略
from explainability.counterfactual import InterventionStrategy
comparator = InterventionStrategy(cf)
strategy_results = comparator.compare_strategies(
    data, learned_scores, topo_scores
)
```

### 4. 韧性评估

```python
from resilience.metrics import ResilienceMetrics
from resilience.assessment import ResilienceAssessor

# 计算韧性指标
metrics = ResilienceMetrics.compute_all(G)

# 完整评估报告
assessor = ResilienceAssessor(model)
report = assessor.assess(data, G, network_id="power_grid_us")
```

## 关键参数调优指南

| 参数 | 默认值 | 调优范围 | 影响 |
|------|--------|---------|------|
| `hidden_channels` | 64 | 32-256 | 模型容量 |
| `num_layers` | 3 | 2-4 | 感受野大小 |
| `dropout` | 0.3 | 0.1-0.5 | MC Dropout质量 |
| `mc_samples` | 30 | 10-100 | 不确定性估计精度 |
| `topo_weight` | 0.4 | 0.2-0.6 | 融合偏好 |
| `gnnexplainer_epochs` | 200 | 100-500 | 解释质量 vs 时间 |

## 如果在CPU上运行

BILGR训练在CPU上也完全可行（约半天），只修改配置:

```python
cfg.experiment.device = "cpu"
cfg.experiment.epochs = 100  # 减少训练轮数
cfg.model.mc_samples = 10    # 减少MC采样
```

## 真实数据加载

```python
# IEEE多域基础设施
data = pipeline.run(
    "ieee_multidomain",
    filepath="data/raw/community_hfg_model.graphml"
)

# 电网
data = pipeline.run(
    "power_grid_mat",
    filepath="data/raw/USpowerGrid.mat"
)
```

## 可能的改进方向 (后续扩展)

### 短期改进 (论文修回用)
1. **超参敏感性分析**: 对topo_weight, dropout做更细致的网格搜索
2. **更多真实数据集验证**: 加入Shelby County, Halmstad数据
3. **更多基线方法**: 加入GradCAM, Integrated Gradients

### 中期改进 (下一篇论文)
1. **异构图扩展**: 支持多类型节点/边的可解释性
2. **时序韧性**: 引入时间维度，分析韧性随时间演化
3. **多目标优化**: 同时优化韧性+成本+公平性

### 长期方向
1. **数字孪生集成**: 实时数据驱动的韧性评估
2. **大语言模型辅助**: LLM生成自然语言韧性报告
3. **联邦学习**: 跨组织的基础设施韧性协作评估
