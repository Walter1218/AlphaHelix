# AlphaHelix vs Qlib 对比分析与改进方案

## 一、系统对比

### 1.1 数据层

| 维度 | Qlib | AlphaHelix | 评价 |
|---|---|---|---|
| 数据源 | Yahoo Finance | Tushare + AkShare | ✅ 我们更好（A股专用） |
| 特征集 | Alpha158/360 标准化 | 自定义特征 | ⚠️ 缺少标准化 |
| 数据频率 | 日频 + 分钟频 | 日频 | ⚠️ Qlib 更丰富 |
| 数据更新 | 自动增量更新 | 手动更新 | ⚠️ 需要自动化 |
| 数据质量 | 有健康检查脚本 | 无 | ⚠️ 需要添加 |

### 1.2 特征工程

| 维度 | Qlib | AlphaHelix | 评价 |
|---|---|---|---|
| 特征数量 | 158/360 个 | 64 个 | ⚠️ 偏少 |
| 特征类型 | kbar + price + rolling | 自定义 | ⚠️ Qlib 更系统 |
| 特征归一化 | 除以当前价格 | StandardScaler | ⚠️ Qlib 更合理 |
| 行业中性化 | 无 | 有（行业内排名） | ✅ 我们更好 |
| 截面排名 | RANK 操作 | 有 | ✅ 相当 |

### 1.3 模型层

| 维度 | Qlib | AlphaHelix | 评价 |
|---|---|---|---|
| 模型种类 | 20+ 种 | 3 种 | ⚠️ 偏少 |
| 模型目标 | 回归（MSE） | 回归（对数收益） | ✅ 相当 |
| 超参数调优 | 配置文件驱动 | 手动调优 | ⚠️ 需要自动化 |
| 模型集成 | DoubleEnsemble | Ridge + XGB | ⚠️ 更简单 |
| 模型保存 | 自动保存 | 手动保存 | ⚠️ 需要自动化 |

### 1.4 评估层

| 维度 | Qlib | AlphaHelix | 评价 |
|---|---|---|---|
| 信号评估 | IC, ICIR, Rank IC, Rank ICIR | IC, MeanIC, ICIR | ⚠️ 缺少 Rank IC |
| 组合评估 | Annualized Return, IR, Max DD | 服务胜率, 累计超额, 夏普 | ✅ 相当 |
| 评估粒度 | 每日每只股票 | 每日每只股票 | ✅ 相当 |
| 评估自动化 | 自动化 | 手动 | ⚠️ 需要自动化 |

### 1.5 策略层

| 维度 | Qlib | AlphaHelix | 评价 |
|---|---|---|---|
| 策略类型 | TopkDropoutStrategy | 质量因子召回 + Top-N | ✅ 我们更灵活 |
| 换仓规则 | 每日最多换 n_drop 只 | 每日全量换仓 | ⚠️ 需要限制换仓 |
| 行业约束 | 无 | 有（质量因子） | ✅ 我们更好 |
| 交易成本 | 有 | 无 | ⚠️ 需要添加 |

### 1.6 回测层

| 维度 | Qlib | AlphaHelix | 评价 |
|---|---|---|---|
| 回测框架 | 完整回测引擎 | 简化回测 | ⚠️ 需要完善 |
| 交易成本 | 有 | 无 | ⚠️ 需要添加 |
| 滑点 | 有 | 无 | ⚠️ 需要添加 |
| 涨跌停限制 | 有 | 无 | ⚠️ 需要添加 |
| 基准对比 | 有 | 有 | ✅ 相当 |

### 1.7 工程化

| 维度 | Qlib | AlphaHelix | 评价 |
|---|---|---|---|
| 配置驱动 | YAML 配置文件 | 环境变量 + 命令行 | ⚠️ 需要配置文件 |
| 自动化 | qrun 一键运行 | 手动运行 | ⚠️ 需要自动化 |
| 模块化 | 高度模块化 | 中度模块化 | ⚠️ 需要改进 |
| 文档 | 完善 | 完善 | ✅ 相当 |
| 测试 | 有 | 无 | ⚠️ 需要添加 |

---

## 二、我们的优势

| 项目 | 说明 |
|---|---|
| ✅ 行业中性化特征 | 行业内排名，Qlib 没有 |
| ✅ 质量因子召回 | 过滤垃圾股，Qlib 没有 |
| ✅ 模型版本管理 | model_registry.py，Qlib 没有 |
| ✅ 风险管理 | 动态仓位、回撤控制，Qlib 没有 |
| ✅ A股专用数据 | Tushare + AkShare，比 Qlib 的 Yahoo Finance 更好 |

---

## 三、改进方案

### 3.1 优先级 P0（必须做）

#### 3.1.1 添加交易成本模型

**问题**：当前回测没有考虑交易成本，结果过于乐观。

**方案**：
```python
# scripts/transaction_cost.py
class TransactionCost:
    def __init__(self, open_cost=0.0005, close_cost=0.0015, min_cost=5):
        self.open_cost = open_cost  # 买入成本 0.05%
        self.close_cost = close_cost  # 卖出成本 0.15%
        self.min_cost = min_cost  # 最低成本 5 元
    
    def calculate(self, amount, is_open=True):
        """计算交易成本"""
        cost_rate = self.open_cost if is_open else self.close_cost
        cost = amount * cost_rate
        return max(cost, self.min_cost)
```

**预期效果**：回测结果更真实。

#### 3.1.2 限制换仓数量

**问题**：当前每日全量换仓，交易成本高。

**方案**：
```python
# scripts/portfolio_strategy.py
class TopkDropoutStrategy:
    def __init__(self, topk=50, n_drop=5):
        self.topk = topk  # 持仓数量
        self.n_drop = n_drop  # 每日最多换仓数量
    
    def generate_signal(self, predictions, current_positions):
        """生成换仓信号"""
        # 1. 按预测分数排序
        ranked = predictions.sort_values('predicted', ascending=False)
        
        # 2. 选择 topk
        target = ranked.head(self.topk)
        
        # 3. 限制换仓数量
        to_sell = [p for p in current_positions if p not in target.index]
        to_buy = [p for p in target.index if p not in current_positions]
        
        # 4. 限制每日换仓
        to_sell = to_sell[:self.n_drop]
        to_buy = to_buy[:self.n_drop]
        
        return to_sell, to_buy
```

**预期效果**：降低交易成本，提高实际收益。

#### 3.1.3 使用标准化特征集

**问题**：当前特征集不够系统，可能遗漏重要特征。

**方案**：实现 Alpha158 风格特征集。
```python
# scripts/alpha158_features.py
class Alpha158Features:
    """Qlib 风格 Alpha158 特征集"""
    
    @staticmethod
    def get_kbar_features(df):
        """K线特征"""
        features = {}
        features['KMID'] = (df['close'] - df['open']) / df['open']
        features['KLEN'] = (df['high'] - df['low']) / df['open']
        features['KMID2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
        features['KUP'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        features['KUP2'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-12)
        features['KLOW'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        features['KLOW2'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-12)
        features['KSFT'] = (2 * df['close'] - df['high'] - df['low']) / df['open']
        features['KSFT2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-12)
        return features
    
    @staticmethod
    def get_rolling_features(df, windows=[5, 10, 20, 30, 60]):
        """滚动特征"""
        features = {}
        for d in windows:
            # ROC (Rate of Change)
            features[f'ROC{d}'] = df['close'].shift(d) / df['close']
            
            # MA (Moving Average)
            features[f'MA{d}'] = df['close'].rolling(d).mean() / df['close']
            
            # STD (Standard Deviation)
            features[f'STD{d}'] = df['close'].rolling(d).std() / df['close']
            
            # BETA (Slope)
            features[f'BETA{d}'] = df['close'].rolling(d).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == d else 0
            ) / df['close']
            
            # MAX (Max High)
            features[f'MAX{d}'] = df['high'].rolling(d).max() / df['close']
            
            # MIN (Min Low)
            features[f'MIN{d}'] = df['low'].rolling(d).min() / df['close']
            
            # RSV (Relative Strength Value)
            features[f'RSV{d}'] = (df['close'] - df['low'].rolling(d).min()) / (
                df['high'].rolling(d).max() - df['low'].rolling(d).min() + 1e-12
            )
            
            # CORR (Correlation)
            features[f'CORR{d}'] = df['close'].rolling(d).corr(np.log(df['volume'] + 1))
            
            # CORD (Correlation of Returns)
            features[f'CORD{d}'] = (df['close'] / df['close'].shift(1)).rolling(d).corr(
                np.log(df['volume'] / df['volume'].shift(1) + 1)
            )
            
            # CNTP (Count Positive)
            features[f'CNTP{d}'] = (df['close'] > df['close'].shift(1)).rolling(d).mean()
            
            # CNTN (Count Negative)
            features[f'CNTN{d}'] = (df['close'] < df['close'].shift(1)).rolling(d).mean()
            
            # SUMP (Sum Positive / RSI)
            features[f'SUMP{d}'] = (
                (df['close'] - df['close'].shift(1)).clip(lower=0).rolling(d).sum() /
                (df['close'] - df['close'].shift(1]).abs().rolling(d).sum() + 1e-12)
            )
            
            # VMA (Volume Moving Average)
            features[f'VMA{d}'] = df['volume'].rolling(d).mean() / (df['volume'] + 1e-12)
            
            # VSTD (Volume Standard Deviation)
            features[f'VSTD{d}'] = df['volume'].rolling(d).std() / (df['volume'] + 1e-12)
        
        return features
```

**预期效果**：特征更系统，可能提升 IC。

---

### 3.2 优先级 P1（应该做）

#### 3.2.1 添加 Rank IC 和 Rank ICIR

**问题**：当前缺少 Rank IC 评估。

**方案**：
```python
# scripts/evaluation.py
def calc_rank_ic(pred, actual):
    """计算 Rank IC"""
    from scipy.stats import spearmanr
    ic, _ = spearmanr(pred, actual)
    return ic if not np.isnan(ic) else 0

def calc_rank_icir(ics):
    """计算 Rank ICIR"""
    return np.mean(ics) / (np.std(ics) + 1e-6)
```

**预期效果**：评估更全面。

#### 3.2.2 尝试更多模型

**问题**：当前只有 Ridge/LGB/XGB，可能错过更好的模型。

**方案**：添加 DoubleEnsemble 和 TRA。
```python
# scripts/models/double_ensemble.py
class DoubleEnsemble:
    """Qlib 的 DoubleEnsemble 模型"""
    
    def __init__(self, base_model='lgb', n_estimators=100):
        self.base_model = base_model
        self.n_estimators = n_estimators
    
    def fit(self, X, y):
        # 1. 训练基础模型
        self.model1 = self._train_base_model(X, y)
        
        # 2. 计算残差
        residuals = y - self.model1.predict(X)
        
        # 3. 训练残差模型
        self.model2 = self._train_base_model(X, residuals)
    
    def predict(self, X):
        return self.model1.predict(X) + self.model2.predict(X)
```

**预期效果**：可能提升 IC。

#### 3.2.3 自动化评估流程

**问题**：当前评估流程手动运行。

**方案**：
```python
# scripts/auto_evaluate.py
class AutoEvaluator:
    """自动化评估器"""
    
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
    
    def run(self):
        # 1. 加载数据
        data = self.load_data()
        
        # 2. 训练模型
        model = self.train_model(data)
        
        # 3. 生成预测
        predictions = model.predict(data)
        
        # 4. 评估信号
        signal_metrics = self.evaluate_signal(predictions)
        
        # 5. 评估组合
        portfolio_metrics = self.evaluate_portfolio(predictions)
        
        # 6. 生成报告
        self.generate_report(signal_metrics, portfolio_metrics)
```

**预期效果**：提高效率。

---

### 3.3 优先级 P2（可以做）

#### 3.3.1 YAML 配置文件驱动

**问题**：当前配置分散在环境变量和命令行。

**方案**：
```yaml
# config/workflow.yaml
data:
  source: tushare
  start_date: 2020-01-01
  end_date: 2026-06-01
  features: alpha158

model:
  type: lightgbm
  params:
    n_estimators: 100
    learning_rate: 0.05
    num_leaves: 31
    max_depth: 6

strategy:
  type: topk_dropout
  topk: 50
  n_drop: 5

backtest:
  start_date: 2024-01-01
  end_date: 2026-06-01
  initial_capital: 100000000
  transaction_cost:
    open: 0.0005
    close: 0.0015
    min: 5
```

**预期效果**：配置更清晰，可复现性更好。

#### 3.3.2 数据质量检查

**问题**：当前没有数据质量检查。

**方案**：
```python
# scripts/data_quality.py
class DataQualityChecker:
    """数据质量检查器"""
    
    def check(self, df):
        issues = []
        
        # 1. 检查缺失值
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"缺失值: {missing[missing > 0].to_dict()}")
        
        # 2. 检查异常值
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            outliers = ((df[col] < q1) | (df[col] > q99)).sum()
            if outliers > 0:
                issues.append(f"{col} 异常值: {outliers}")
        
        # 3. 检查数据连续性
        dates = df['date'].unique()
        date_diff = np.diff(dates)
        if np.max(date_diff) > np.timedelta64(7, 'D'):
            issues.append("数据不连续")
        
        return issues
```

**预期效果**：数据质量更好。

#### 3.3.3 添加测试

**问题**：当前没有测试。

**方案**：
```python
# tests/test_model.py
import unittest

class TestModel(unittest.TestCase):
    def test_ridge_model(self):
        """测试 Ridge 模型"""
        model = Ridge(alpha=1.0)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), 100)
    
    def test_lgb_model(self):
        """测试 LightGBM 模型"""
        model = lgb.LGBMRegressor(n_estimators=10)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), 100)
```

**预期效果**：代码质量更好。

---

## 四、实施计划

### 第一阶段（1-2 周）

| 任务 | 优先级 | 预期效果 |
|---|---|---|
| 添加交易成本模型 | P0 | 回测结果更真实 |
| 限制换仓数量 | P0 | 降低交易成本 |
| 使用标准化特征集 | P0 | 可能提升 IC |

### 第二阶段（2-4 周）

| 任务 | 优先级 | 预期效果 |
|---|---|---|
| 添加 Rank IC 和 Rank ICIR | P1 | 评估更全面 |
| 尝试更多模型 | P1 | 可能提升 IC |
| 自动化评估流程 | P1 | 提高效率 |

### 第三阶段（4-8 周）

| 任务 | 优先级 | 预期效果 |
|---|---|---|
| YAML 配置文件驱动 | P2 | 配置更清晰 |
| 数据质量检查 | P2 | 数据质量更好 |
| 添加测试 | P2 | 代码质量更好 |

---

## 五、预期收益

| 改进项 | 预期 IC 提升 | 预期胜率提升 |
|---|---|---|
| 交易成本模型 | 无 | -2% ~ -5%（更真实） |
| 限制换仓数量 | 无 | +1% ~ +3% |
| 标准化特征集 | +0.01 ~ +0.02 | +2% ~ +5% |
| 更多模型 | +0.005 ~ +0.01 | +1% ~ +3% |
| 自动化评估 | 无 | 无（效率提升） |

**总体预期**：IC 从 0.058 提升到 0.07-0.08，服务胜率从 54.3% 提升到 55-58%。
