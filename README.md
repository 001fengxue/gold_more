# 积存金决策辅助系统

这是一个面向个人黄金积存金的轻量 MVP，用来做行情观察、规则信号、回测评估和风险提示。它不是自动交易系统，也不承诺收益。

## 为什么从零搭建

GitHub 上有不少 XAUUSD/MT5/AI 交易机器人，但多数面向外汇黄金、杠杆和日内交易，和国内银行积存金的人民币报价、买卖价差、定投、赎回规则并不一致。本项目选择重新搭建业务层，同时复用成熟 Python 数据与分析生态。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

如果 AKShare 或上金所接口临时不可用，页面会自动切到演示数据，核心回测功能仍然可运行。

## 命令行回测

```powershell
python scripts\run_backtest.py --source demo
python scripts\run_backtest.py --source akshare --symbol Au99.99
```

## 当前策略

第一版采用规则策略，而不是直接上黑盒模型：

- 趋势：短均线与长均线判断主趋势。
- 回撤：从阶段高点回落到一定幅度时提高买入权重。
- 过热：RSI 偏高且价格远离均线时降低仓位。
- 风控：通过目标仓位、调仓阈值和买卖价差模拟真实交易摩擦。

后续可以加入宏观因子、参数寻优、滚动样本外测试和模型预测，但需要先把“回测是否可信”立住。

## 技术栈与价格口径

技术栈、数据口径和“为什么价格不会实时跳动”的说明见 [docs/TECH_STACK.md](docs/TECH_STACK.md)。
