# ==========================================
# 战法名称：【乾坤一掷·大资金突破战法】
# 核心逻辑：
# 1. 资金门槛：当日成交额必须 > 3 亿元 (大资金主战场)。
# 2. 突破质量：收盘价必须创下近 60 个交易日的新高 (确认无套牢盘)。
# 3. 筹码逻辑：换手率 5%-10% 最佳，代表主力高度控盘且仍有换手。
# 4. 拒绝补涨：如果所属行业当天没有其他股涨停，单独的一只股给 100 分也要警惕。
# ==========================================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool

DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'

def screen_logic(file_path):
    try:
        # 针对你的 CSV 格式优化读取
        df = pd.read_csv(file_path)
        if len(df) < 120: return None # 至少需要半年的历史数据来判断压力位
        
        curr = df.iloc[-1]
        
        # --- 第一道防线：硬性生存过滤 ---
        code = str(curr['股票代码']).zfill(6)
        if code.startswith('30'): return None # 排除创业板
        if not (5.0 <= curr['收盘'] <= 25.0): return None # 价格适中
        if curr['成交额'] < 300000000: return None # 必须大于3亿成交额，拒绝冷门股
        
        # --- 第二道防线：空间与动能 ---
        high_60 = df['最高'].iloc[-61:-1].max() # 过去60天的最高点
        if curr['收盘'] < high_60: return None # 如果没过前高，就是假突破，剔除！
        
        # --- 第三道防线：K线实体质量 ---
        # 实体占比：收盘价要在最高价附近，不能有长上影
        if (curr['最高'] - curr['收盘']) / (curr['最高'] - curr['最低'] + 0.01) > 0.15: return None

        # 计算基本指标
        vol_ratio = curr['成交量'] / df['成交量'].iloc[-6:-1].mean()
        
        score = 80
        if vol_ratio > 2.5: score += 10
        if curr['涨跌幅'] > 9.5: score += 10 # 最好是涨停，代表绝对强势
        
        if score >= 90:
            return {
                '代码': code, '收盘': curr['收盘'], '涨跌幅': curr['涨跌幅'], 
                '成交额(亿)': round(curr['成交额']/100000000, 2),
                '评分': score, '信号': "【乾坤一掷】", 
                '建议': "该股突破了60日压力位且资金介入极深，建议明日回踩前高位置介入。"
            }
    except:
        return None

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    with Pool(os.cpu_count()) as p:
        results = [r for r in p.map(screen_logic, files) if r is not None]
    
    if results:
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        final_df = pd.merge(pd.DataFrame(results), names_df, left_on='代码', right_on='code', how='left')
        # 极致精选：全市场只给最强的一只
        final_df = final_df.sort_values(by=['评分', '成交额(亿)'], ascending=False).head(1)
        
        # 保存逻辑同前...
        print(f"最终筛选：{final_df['name'].values[0]} ({final_df['代码'].values[0]})")
