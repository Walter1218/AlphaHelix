#!/bin/bash
# 每日选股报告定时任务
# 每天 19:00 执行

# 切换到项目目录
cd /Users/onetwo/Documents/trae_projects/AlphaHelix

# 加载环境变量
export $(grep -v '^#' .env | xargs)

# 1. 更新数据并生成预测
echo "[$(date)] 开始更新数据..."
/usr/bin/python3 scripts/update_and_predict.py >> /Users/onetwo/Documents/trae_projects/AlphaHelix/logs/daily_report.log 2>&1

# 2. 运行每日报告（验证 + 选股）
echo "[$(date)] 开始生成报告..."
/usr/bin/python3 scripts/daily_report.py >> /Users/onetwo/Documents/trae_projects/AlphaHelix/logs/daily_report.log 2>&1

echo "[$(date)] 完成"
