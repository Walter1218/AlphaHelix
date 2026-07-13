#!/bin/bash
# 每日选股报告定时任务
# 每天 19:00 执行

# 切换到项目目录
cd /Users/onetwo/Documents/trae_projects/AlphaHelix

# 加载环境变量
export $(grep -v '^#' .env | xargs)

# 运行报告
/usr/bin/python3 scripts/daily_report.py >> /Users/onetwo/Documents/trae_projects/AlphaHelix/logs/daily_report.log 2>&1
