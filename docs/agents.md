# AlphaHelix 智能体设计

> **核心框架**：基于 [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) (19.9k stars) 的方法论重构。

## 0. 核心原则（强制）

### 0.1 过程即优势

> **"The process is the edge, not the model."**

量化交易的核心不是模型复杂度，而是**严谨的过程**。每次实验、每次决策都必须遵循以下框架。

### 0.2 证据边界（Evidence Boundary）

**强制要求**：严格分离**调优期**和**评估期**。

```
数据时间线:
[-------- 调优期 --------][--- 评估期 ---]
     用于选特征/调参数       用于最终评估
```

**规则**：
1. 调优期的数据**绝不能**用于评估
2. 评估期的数据**绝不能**用于调优
3. 评估期只运行**一次**，不能反复测试

**违规示例**：
- ❌ 用 2023-2025 数据调优，再用 2023-2025 数据评估
- ❌ 发现评估结果不好，回去改参数再评估
- ✅ 用 2020-2023 调优，用 2024-2026 评估

### 0.3 数据泄漏红线（强制）

**任何形式的数据泄漏都是不可接受的。**

#### 禁止行为

| 类型 | 说明 | 示例 |
|---|---|---|
| **时间穿越** | 使用未来数据 | 用 T+1 的价格预测 T |
| **特征泄漏** | 特征包含目标信息 | 用 tomorrow_return 作为特征 |
| **样本泄漏** | 训练/测试数据重叠 | 同一天数据出现在训练和测试集 |
| **选择偏差** | 用全量数据做选择 | 用全量 IC 选特征再 walk-forward |
| **前瞻偏差** | 用未来信息做决策 | 用测试期的统计量调参 |

#### 检查清单（每次实验必须）

- [ ] 特征是否只使用历史数据？
- [ ] 目标变量是否只使用未来数据？
- [ ] 特征选择是否在 walk-forward 每期内完成？
- [ ] 训练/测试是否有时间重叠？
- [ ] 评估是否只在测试集上进行？

### 0.4 实验记录（强制）

**每次实验必须记录到 `docs/experiment_log.md`。**

#### 记录格式

```markdown
## 实验 N：实验名称（日期）

### 实验设计
- **目标**：解决什么问题
- **假设**：预期效果

### 数据配置
- **数据范围**：YYYY-MM-DD ~ YYYY-MM-DD
- **训练窗口**：N 个月
- **测试窗口**：N 个月
- **Purge gap**：N 天

### 特征配置
- **特征数量**：N 个
- **特征选择方法**：IC / AUC / 其他
- **选择是否在 walk-forward 内完成**：是/否

### 模型配置
- **模型类型**：Ridge / LightGBM / 其他
- **超参数**：列出关键参数

### 实验结果
| 指标 | 值 |
|---|---|
| IC | X.XXXX |
| ICIR | X.XX |
| 胜率 | XX.X% |
| 收益 | +XX.X% |

### 数据泄漏检查
- [ ] 特征无未来数据
- [ ] 目标无历史数据
- [ ] 选择在 walk-forward 内
- [ ] 评估在测试集上

### 结论
简要总结
```

---

## 1. Walk-Forward 验证框架（强制）

### 1.1 标准 Walk-Forward 流程

```
时间线: [--- 训练期 ---][-- Purge --][-- 测试期 --]
                      ↑            ↑
                   不含数据      不含数据
```

**Purge gap**：训练期和测试期之间的间隔，防止数据泄漏。
**Embargo**：测试期之后的间隔，防止信息泄漏。

### 1.2 实现模板

```python
def walk_forward(df, months, train_w=6, purge=1, test_w=1):
    """
    Walk-forward 验证框架
    
    Args:
        df: 数据
        months: 月份列表
        train_w: 训练窗口（月）
        purge: Purge gap（月）
        test_w: 测试窗口（月）
    """
    results = []
    
    for i in range(len(months)):
        # 训练期
        train_end = i + train_w
        if train_end >= len(months):
            break
        train_months = months[i:train_end]
        
        # Purge gap
        purge_start = train_end
        purge_end = purge_start + purge
        if purge_end >= len(months):
            break
        
        # 测试期
        test_start = purge_end
        test_end = test_start + test_w
        if test_end >= len(months):
            break
        test_month = months[test_start]
        
        # 获取数据
        train_df = df[df['ym'].isin(train_months)]
        test_df = df[df['ym'] == test_month]
        
        # 在训练集上做所有调优
        # ...
        
        # 在测试集上评估
        # ...
        
        results.append(...)
    
    return results
```

### 1.3 关键规则

1. **特征选择必须在训练集内完成**
2. **超参数调优必须在训练集内完成**
3. **模型训练只用训练集数据**
4. **评估只用测试集数据**

---

## 2. 特征工程规范

### 2.1 五大特征家族

参考 ML4T 框架，系统化构建特征：

| 家族 | 说明 | 示例 |
|---|---|---|
| **动量** | 价格趋势 | mom_5, mom_20, mom_60 |
| **反转** | 均值回归 | reversal_score |
| **波动率** | 价格波动 | volatility_20 |
| **流动性** | 交易活跃度 | liquidity, amount_ratio |
| **基本面** | 财务指标 | roe, dv_ratio, ep, bp |

### 2.2 特征选择规范

**必须在 walk-forward 每期内完成特征选择。**

```python
# 正确：在每期内用训练数据选择特征
for i in range(len(months)):
    train_df = df[df['ym'].isin(train_months)]
    
    # 用训练数据计算 IC
    ics = {}
    for col in feature_cols:
        ic = calc_ic(train_df[col], train_df['target'])
        ics[col] = ic
    
    # 选择正 IC 特征
    selected = [c for c, ic in ics.items() if ic > 0][:n]
    
    # 训练模型
    # ...
```

### 2.3 特征工程规范

**禁止**：
- ❌ 用全量数据计算统计量
- ❌ 用未来数据归一化
- ❌ 用测试集信息做特征选择

**允许**：
- ✅ 用训练数据计算统计量
- ✅ 用历史数据归一化
- ✅ 用训练集信息做特征选择

---

## 3. 模型规范

### 3.1 模型选择

| 模型 | 适用场景 | 优势 | 劣势 |
|---|---|---|---|
| Ridge | 小数据集 | 稳定、不易过拟合 | 线性假设 |
| LightGBM | 大数据集 | 捕获非线性 | 易过拟合 |
| XGBoost | 大数据集 | 正则化强 | 训练慢 |

### 3.2 模型评估指标

| 指标 | 定义 | 说明 |
|---|---|---|
| **IC** | Spearman(pred, actual) | 预测与实际的相关性 |
| **ICIR** | mean(IC) / std(IC) | IC 的稳定性 |
| **胜率** | Top-N 中正收益比例 | 选股准确率 |
| **累计收益** | Top-N 组合的累计收益 | 组合盈利能力 |
| **夏普比率** | 收益/波动率 | 风险调整收益 |
| **最大回撤** | 最大累计亏损 | 风险控制 |

### 3.3 模型训练规范

```python
# 正确：在 walk-forward 每期内训练
for i in range(len(months)):
    train_df = df[df['ym'].isin(train_months)]
    test_df = df[df['ym'] == test_month]
    
    # 特征选择（用训练数据）
    selected = select_features(train_df)
    
    # 训练模型（用训练数据）
    model = train_model(train_df[selected])
    
    # 预测（用测试数据）
    predictions = model.predict(test_df[selected])
    
    # 评估（用测试数据）
    metrics = evaluate(predictions, test_df['actual'])
```

---

## 4. 交易成本与风险管理

### 4.1 交易成本模型

| 成本项 | 费率 | 说明 |
|---|---|---|
| 佣金 | 万三 | 买卖都收 |
| 印花税 | 万五 | 卖出收 |
| 过户费 | 十万分之一 | 买卖都收 |
| 滑点 | 千一 | 估算 |

### 4.2 换仓限制

| 参数 | 值 | 说明 |
|---|---|---|
| 每日最多换仓 | 5 只 | 降低交易成本 |
| 单只最大仓位 | 15% | 分散风险 |
| 单行业最大仓位 | 40% | 行业分散 |

### 4.3 风险管理

| 指标 | 阈值 | 动作 |
|---|---|---|
| 回撤 > 15% | 警告 | 关注 |
| 回撤 > 25% | 降仓 | 减少 20% 仓位 |
| 回撤 > 35% | 停止 | 清仓观望 |

---

## 5. 服务链路规范

### 5.1 两阶段架构

```
全市场 ~5300 只
    ↓ 第一阶段：召回（质量因子过滤）
召回池 ~500 只
    ↓ 第二阶段：排序（模型打分）
Top-N 输出
```

### 5.2 召回策略

**质量因子召回**：
- ROE > 0
- 资金流入 > 0

**目的**：排除垃圾股，不是选择好股。

### 5.3 指标体系

| 阶段 | 指标 | 定义 |
|---|---|---|
| **服务链路** | 服务胜率 | Top-N 中超额收益>0 的比例 |
| **服务链路** | 服务累计超额 | Top-N 组合的累计超额收益 |
| **模型阶段** | IC | 模型预测与实际收益的相关性 |
| **模型阶段** | ICIR | IC 的稳定性 |

---

## 6. 当前最优配置

> **最后更新**：2026-07-16

| 参数 | 值 |
|---|---|
| 模型 | Ridge alpha=1.0 |
| 特征数 | 20（正 IC 选择） |
| 训练窗口 | 3 个月 |
| 目标 | 5 天均值超额收益 |
| 召回 | 质量因子（ROE>0 & 资金流入>0） |
| 持仓 | Top-10 |
| **IC** | **0.01-0.04** |
| **胜率** | **50-56%** |
| **收益** | 取决于市场 |

### 真实性能评估

| 指标 | 真实水平 | 说明 |
|---|---|---|
| IC | 0.01-0.04 | 股票预测本身就难 |
| 胜率 | 50-56% | 略高于随机 |
| 夏普 | 不稳定 | 取决于市场环境 |

**注意**：IC 0.01-0.04 是股票预测的真实水平。之前看到的高 IC 是短期波动或数据问题。

---

## 7. 文件结构

```
AlphaHelix/
├── scripts/
│   ├── daily_report.py          # 每日报告
│   ├── incremental_trainer.py   # 增量训练
│   ├── feishu_bot.py            # 飞书推送
│   ├── transaction_cost.py      # 交易成本
│   ├── portfolio_strategy.py    # 组合策略
│   └── risk_management.py       # 风险管理
├── docs/
│   ├── AGENTS.md                # 本文档
│   ├── experiment_log.md        # 实验记录
│   └── qlib_comparison.md       # Qlib 对比
├── memory/
│   ├── models/                  # 模型文件
│   ├── predictions/             # 预测结果
│   └── fundamental/             # 基本面数据
└── .env                         # 环境变量
```

---

## 8. 常用命令

```bash
# 每日报告
python scripts/daily_report.py

# 增量训练
python scripts/incremental_trainer.py --incremental

# 飞书推送测试
python scripts/feishu_bot.py --push "测试消息"

# 回测验证
python scripts/backtest_with_cost.py
```
