# 实验班科研拔尖训练项目

本仓库包含 Graph-LassoNet 相关实验代码。仓库仅保留代码与运行依赖说明，不包含论文、汇报材料、数据集、日志或实验结果文件。本项目获得了优秀项目结项。

## 代码结构

- `src/lapreg_lassonet/`：核心 Python 包。
  - `config.py`：实验配置与参数定义。
  - `data/`：数据集加载、图结构与拉普拉斯矩阵处理。
  - `models/`：图正则化 LassoNet 模型实现。
  - `train/`：模型训练流程。
  - `eval/`：评价指标。
  - `utils/`：路径、输入输出等通用工具。
- `scripts/`：数据准备、图先验构建、训练、消融实验、稳定性分析、结果汇总与绘图脚本。
- `legacy/`：早期原型与历史实验脚本，保留用于参考。
- 根目录 Python 脚本：若干 BRCA/Reactome 相关的独立分析与对比脚本。
- `requirements.txt`：运行代码所需的 Python 依赖。

## 不纳入仓库的内容

- `datasets/`：外部数据与中间数据文件。
- `results/`：实验输出结果。
- `logs/`：运行日志。
- `presentation_beamer/`、`学术训练结项论文/`、`学术训练结项论文.zip`：presentation 与论文相关材料。

