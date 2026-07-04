# 基于 LLM 的智能选股调研

> 调研日期：2026-07-04  
> 来源：GitHub 高 Star 项目、FinLLM 论文、行业实践

---

## 1. 高 Star LLM+金融项目速览

| 项目 | Stars | 定位 | LLM 用法 |
|---|---|---|---|
| [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | 20.8k | 开源金融大模型 | 金融情感分析、新闻/报告摘要、股价走势预测（Forecaster）、RAG 检索增强 |
| [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | 13.8k | 自动化量化 R&D Agent | LLM 读取论文/财报 → 自动提取因子公式 → 代码实现 → 回测验证 → 迭代优化 |
| [OpenBB-finance/OpenBB](https://github.com/OpenBB-finance/OpenBB) | 70k | 开源数据平台 | 统一数据接口 + AI Copilot/Agent，支持 LLM 做研究问答和报告生成 |
| [FinRL-X / FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL) | 15.6k | 金融强化学习 | 用 LLM 做策略推理、多 Agent 辩论、自然语言生成交易逻辑 |
| [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab) | 4.9k | 金融 ML 工具箱 | 虽非 LLM，但其元标签(meta-labeling)理念常被 LLM 用于生成下注信号 |

---

## 2. LLM 在选股中的 6 种用法

### 2.1 情感/事件信号提取（最成熟）

- **输入**：新闻、公告、财报、社交媒体、研报、政策文件。
- **输出**：情感分数、事件类型、重要性、行业影响。
- **落地**：把 LLM 输出作为**因子**喂给量化模型（LightGBM/XGBoost）。
- **代表**：FinGPT 情感模型、FinBERT、FinGPT-RAG。

### 2.2 财报/研报信息抽取

- **输入**：年报、季报、业绩预告、业绩说明会纪要、卖方研报。
- **输出**：
  - 业绩超预期/低于预期判断；
  - 管理层语调（乐观/谨慎）；
  - 行业景气度、资本开支计划、风险提示。
- **落地**：构建事件驱动因子，弥补 Tushare `forecast`/`express` 覆盖不足。
- **代表**：RD-Agent 财报因子抽取、OpenBB 研报分析。

### 2.3 因子/策略代码生成（前沿）

- **输入**：研究论文、研报段落、人类想法。
- **输出**：可运行的因子计算公式或模型代码。
- **流程**：
  1. LLM 提取公式/逻辑；
  2. 自动生成 Python 代码；
  3. 在 Qlib/AlphaHelix 回测中验证；
  4. 根据 IC/回测结果反馈，迭代修改。
- **代表**：Microsoft RD-Agent（R&D-Agent-Quant）。

### 2.4 RAG 研究助手

- **输入**：历史报告、市场数据、知识库。
- **输出**：针对当前市场环境的投资逻辑、相似案例、风险提醒。
- **落地**：给选股决策提供**定性依据**，但不直接决定买卖。
- **代表**：FinGPT-RAG、OpenBB Workspace AI。

### 2.5 多 Agent 决策/辩论

- **架构**：
  - 宏观分析师 Agent：判断市场 regime；
  - 行业分析师 Agent：评估行业景气度；
  - 基本面分析师 Agent：解读个股财报；
  - 技术分析师 Agent：看量价形态；
  - 风控 Agent：检查集中度、止损、叙事风险。
- **落地**：多 Agent 输出综合为**置信度或权重**，再与量化信号融合。
- **代表**：FinRL-X 多 Agent、一些新兴的 AI Hedge Fund 项目。

### 2.6 直接用 LLM 预测价格/方向（不推荐）

- 常见演示："预测下周某股票涨跌"。
- **问题**：
  - LLM 容易 hallucinate，对价格没有数值敏感性；
  - 训练数据可能包含未来信息，存在 lookahead；
  - 无法解释、不可复现、成本高。
- **结论**：不建议把 LLM 当作唯一预测器。

---

## 3. 行业共识：LLM 应该做什么、不应该做什么

| 适合做 | 不适合做 |
|---|---|
| 从非结构化文本中提取结构化信号 | 直接预测未来价格/方向 |
| 生成因子代码并自动回测验证 | 替代确定性收益计算 |
| 提供定性研究与风险提醒 | 在高频场景中做实时决策 |
| 多 Agent 辩论产生投资逻辑 | 无约束地给出买卖建议 |
| 作为特征工程/信号增强层 | 作为唯一的选股依据 |

---

## 4. 对 AlphaHelix 的启示

### 4.1 我们当前 LLM 用法的问题

- 当前 `alpha-analyst` 已经用 LLM 做选股综合决策，但**量化模型信号弱**，导致 LLM 在“无米之炊”上强行说理。
- `memory_search` 故障导致 RAG 未启用，损失了历史经验复用。
- LLM 输出缺乏与量化信号的明确融合机制。

### 4.2 建议的 LLM 增强路径

1. **情感/事件因子层**
   - 接入新闻、公告、研报，用 LLM 输出每只股票的**情感得分**和**事件标签**。
   - 作为新特征加入 LightGBM/XGBoost 预测模型。
   - 关键：必须按 `trade_date` 切分，只能用当日及之前已公开的信息。

2. **财报深度解读**
   - 对业绩预告、业绩快报、年报摘要做 LLM 解析。
   - 输出：超预期概率、管理层语调、行业对比、风险点。
   - 与 `forecast`/`express` 数据互补。

3. **因子代码生成 Agent**
   - 参考 RD-Agent，让 LLM 从研报/论文中提出因子假设，生成代码，跑回测。
   - 通过 IC/IR 自动筛选有效因子，避免人工试错。

4. **RAG 研究助手**
   - 修复 `memory_search` 后，让 LLM 在选股前检索历史相似市场环境。
   - 输出：历史案例、教训、当前应关注的行业/风险。

5. **多 Agent 组合风控**
   - 量化模型给出候选池和分数；
   - LLM Agent 检查：
     - 行业叙事风险（如监管、政策变化）；
     - 高杠杆/高波动叙事；
     - 业绩暴雷风险；
   - LLM 可以**否决**或**降低权重**，但不应凭空选股。

### 4.3 最小可行下一步

1. 先让底层量化模型有信号（LightGBM/XGBoost + IC/IR 验证）。
2. 再接入 LLM 情感/事件因子，看是否能提升 IC。
3. 再启用 RAG 和多 Agent 风控层。

---

## 5. 参考资源

- [FinGPT 论文与模型](https://github.com/AI4Finance-Foundation/FinGPT)
- [RD-Agent: Quant Factor & Model Co-optimization](https://github.com/microsoft/RD-Agent)
- [OpenBB 文档](https://docs.openbb.co/)
- [FinRL-X 多 Agent 架构](https://github.com/AI4Finance-Foundation/FinRL)
- [FinBERT: Financial Sentiment Analysis](https://huggingface.co/ProsusAI/finbert)
