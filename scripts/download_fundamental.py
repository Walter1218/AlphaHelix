"""
批量下载 Tushare 基本面数据

按股票列表批量下载，使用多线程加速。

用法：
    python download_fundamental.py
"""
import sys
import os
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tushare as ts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')
OUTPUT_DIR = 'memory/fundamental'
MAX_WORKERS = 8  # 并发数


def get_stock_list():
    """获取股票列表"""
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    return df


def download_stock_data(ts_code: str, api_name: str, start_date: str, end_date: str):
    """下载单只股票数据"""
    pro = ts.pro_api(TUSHARE_TOKEN)
    try:
        api_func = getattr(pro, api_name)
        df = api_func(ts_code=ts_code, start_date=start_date, end_date=end_date)
        return df
    except Exception as e:
        return None


def download_api(api_name: str, stock_list: list, start_date: str, end_date: str):
    """批量下载某个API的数据"""
    output_path = os.path.join(OUTPUT_DIR, f'{api_name}.parquet')
    
    logger.info(f"下载 {api_name}: {len(stock_list)} 只股票")
    
    all_data = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_stock_data, ts_code, api_name, start_date, end_date): ts_code
            for ts_code in stock_list
        }
        
        for future in as_completed(futures):
            ts_code = futures[future]
            completed += 1
            
            if completed % 500 == 0:
                logger.info(f"  进度: {completed}/{len(stock_list)}")
            
            try:
                df = future.result()
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:
                pass
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result.drop_duplicates(inplace=True)
        result.to_parquet(output_path, index=False)
        logger.info(f"  保存成功: {output_path}, {len(result)} 行")
        return result
    else:
        logger.warning(f"  {api_name} 无数据")
        return None


def main():
    logger.info("=" * 50)
    logger.info("开始下载 Tushare 基本面数据")
    logger.info("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取股票列表
    logger.info("获取股票列表...")
    stock_df = get_stock_list()
    stock_list = stock_df['ts_code'].tolist()
    logger.info(f"股票数量: {len(stock_list)}")
    
    # 日期范围（最近3年）
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
    logger.info(f"日期范围: {start_date} ~ {end_date}")
    
    # 下载数据
    apis = [
        'fina_indicator',
        'income',
        'balancesheet',
        'cashflow',
        'forecast',
        'express',
        'dividend',
        'fina_mainbz',
    ]
    
    results = {}
    for api in apis:
        result = download_api(api, stock_list, start_date, end_date)
        if result is not None:
            results[api] = len(result)
    
    # 显示下载结果
    logger.info("\n" + "=" * 50)
    logger.info("下载结果:")
    logger.info("=" * 50)
    for api, count in results.items():
        logger.info(f"  {api}: {count} 行")
    
    # 保存股票列表
    stock_df.to_parquet(os.path.join(OUTPUT_DIR, 'stock_list.parquet'), index=False)
    logger.info(f"\n股票列表已保存: {len(stock_df)} 只")


if __name__ == "__main__":
    main()
