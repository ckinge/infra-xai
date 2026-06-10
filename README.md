# Infra-XAI: Explainable Bayesian GNN for Critical Infrastructure Resilience Assessment

> **论文方向**: 基于可解释贝叶斯图神经网络的多维度关键基础设施韧性评估  
> **投稿目标**: MDPI *Systems* 专刊 "Multi-Dimensional Resilience Assessment of Interdependent Critical Infrastructure Systems" (截稿: 2026-08-31)  
> **核心创新**: 多粒度可解释性 + 贝叶斯不确定性量化 + 反事实韧性提升

## 快速开始

```bash
# 环境安装
pip install -r requirements.txt

# 运行完整实验流水线
python experiments/run_all.py
```

## 项目结构

```
infra-xai/
├── config.py                  # 全局配置
├── requirements.txt           # 依赖
├── data/                      # 数据加载与预处理
│   ├── dataset_loader.py      # 多源数据集加载
│   ├── graph_builder.py       # 异构图构建
│   └── preprocess.py          # 预处理流水线
├── models/                    # 模型层
│   ├── bilgr.py               # BILGR 贝叶斯GNN（核心复现）
│   ├── graphsage_encoder.py   # GraphSAGE 编码器
│   └── bayesian_wrapper.py    # MC Dropout + MAP 估计
├── explainability/            # 可解释性模块（核心创新）
│   ├── explainer_factory.py   # 可解释器工厂
│   ├── multi_granular.py      # 多粒度解释
│   ├── topology_fusion.py     # 拓扑-学习融合
│   └── counterfactual.py     # 反事实推理
├── resilience/                # 韧性评估
│   ├── metrics.py             # 韧性指标
│   ├── simulation.py          # 级联故障模拟
│   └── assessment.py          # 评估框架
├── experiments/               # 实验脚本
│   ├── exp1_baseline.py       # 可解释性方法比较
│   ├── exp2_multigranular.py  # 多粒度解释质量
│   ├── exp3_counterfactual.py # 反事实韧性提升
│   ├── exp4_ablation.py       # 消融实验
│   ├── exp5_robustness.py     # 鲁棒性分析
│   └── run_all.py             # 一键运行
├── visualization/             # 可视化
│   ├── plot_explanations.py
│   └── plot_metrics.py
└── notebooks/
    └── demo.ipynb
```

## 核心创新

1. **多粒度可解释性**: 节点级 / 边级 / 子图级三层解释
2. **贝叶斯不确定性感知解释**: 高不确定性→低可信度标注
3. **拓扑-学习融合框架**: 传统网络科学指标 ⇄ GNN解释 交叉验证
4. **反事实韧性提升**: "加固节点X → 级联故障风险降低Y%"

## 数据集

- IEEE 多域基础设施网络 (GraphML, DOI: 10.21227/2m92-9f70)
- Kostia-Zuev 电网拓扑 (北美/欧洲, Zenodo)
- 合成 scale-free / small-world 网络（参数化生成）

## 引用

```
Munikoti, S., Das, L., & Natarajan, B. (2021).
Bayesian Graph Neural Network for Fast identification of critical nodes
in Uncertain Complex Networks. IEEE SMC 2021.
```
