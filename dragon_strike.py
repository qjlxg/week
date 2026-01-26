import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool

# ==========================================
# 战法备注：【乾坤一掷·大资金共振突破战法】
# 1. 核心过滤：成交额 > 3 亿 (真金白银堆出来的突破才是真的)。
# 2. 空间逻辑：收盘价必须创近 60 个交易日最高 (上方无套牢盘)。
# 3. 质量逻辑：拒绝长上影线 (收盘价需在全天波动的前 15% 范围内)。
# 4. 筹码逻辑：换手率 3%-15% (主力控盘良好)。
# 5. 排除标准：排除 ST、创业板 (30开头)、价格不在 5-25 元区间。
# ==========================================

DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'

def calculate_indicators(df):
    """计算核心技术指标"""
    close = df['收盘']
    # 均线系统
    df['MA5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    # RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/(loss + 0.00001)))
    return df

def screen_logic(file_path):
    try:
        # 读取 CSV，兼容制表符和逗号
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sep = '\t' if '\t' in first_line else ','
        df = pd.read_csv(file_path, sep=sep)
        
        if len(df) < 65: return None
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        code = str(curr['股票代码']).zfill(6)
        
        # --- 严苛过滤逻辑 ---
        # 1. 基础排除：创业板、ST、价格区间
        if code.startswith('30') or 'ST' in file_path: return None
        if not (5.0 <= curr['收盘'] <= 25.0): return None
        
        # 2. 资金实力：成交额必须 > 3亿
        if curr['成交额'] < 300000000: return None
        
        # 3. 空间阻力：必须突破过去 60 个交易日的最高价压力
        high_60 = df['最高'].iloc[-61:-1].max()
        if curr['收盘'] < high_60: return None
        
        # 4. K 线实体：拒绝冲高回落，上影线必须短
        shadow_ratio = (curr['最高'] - curr['收盘']) / (curr['最高'] - curr['最低'] + 0.01)
        if shadow_ratio > 0.15: return None
        
        # 5. 换手率：3% 到 15% 之间
        if not (3.0 <= curr['换手率'] <= 15.0): return None
        
        # --- 计算技术得分 ---
        df = calculate_indicators(df)
        curr_idx = df.iloc[-1]
        
        score = 80
        # 加分项：倍量起爆
        vol_ratio = curr['成交量'] / df['成交量'].iloc[-6:-1].mean()
        if vol_ratio > 2.0: score += 10
        # 加分项：RSI进入强势区
        if curr_idx['RSI6'] > 80: score += 10
        
        if score >= 90:
            return {
                '代码': code,
                '收盘': curr['收盘'],
                '涨跌幅': curr['涨跌幅'],
                '成交额(亿)': round(curr['成交额']/100000000, 2),
                'RSI6': round(curr_idx['RSI6'], 1),
                '评分': score,
                '信号': "【乾坤一掷·极品】",
                '操作建议': "突破60日高点压力，大资金深度介入。建议明日观察，若不破今日最高价的一半可分批布局。"
            }
    except Exception:
        return None

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # 扫描并筛选
    with Pool(os.cpu_count()) as p:
        results = [r for r in p.map(screen_logic, files) if r is not None]
    
    if results:
        # 匹配名称
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        names_df['code'] = names_df['code'].str.zfill(6)
        
        final_df = pd.DataFrame(results)
        final_df = pd.merge(final_df, names_df[['code', 'name']], left_on='代码', right_on='code', how='left')
        
        # 宁缺毋滥：按评分和成交额排序，只取前 1-2 只
        final_df = final_df.sort_values(by=['评分', '成交额(亿)'], ascending=False).head(2)
        
        # 结果持久化
        now = datetime.now()
        folder = now.strftime('%Y%m')
        os.makedirs(folder, exist_ok=True)
        filename = f"dragon_strike_{now.strftime('%Y%m%d_%H%M')}.csv"
        
        output_cols = ['代码', 'name', '收盘', '涨跌幅', '成交额(亿)', '评分', '信号', '操作建议']
        final_df[output_cols].to_csv(f"{folder}/{filename}", index=False, encoding='utf_8_sig')
        print(f"全自动复盘完毕，精选标的：{final_df['name'].tolist()}")
    else:
        print("今日无符合'乾坤一掷'战法标的，建议空仓复盘。")
