import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool

# ==========================================
# 战法备注：【潜龙出海·缩量回撤起爆战法】
# 核心逻辑：
# 1. 拒绝追高：涨幅超过 7% 的直接踢出。我们要找的是“含苞待放”的。
# 2. 异动基因：过去 5-10 天内必须出现过至少一次大阳线 (涨幅>6%)，证明有主力。
# 3. 缩量回踩：当前的成交量必须是异动当天的 50% 以下 (洗盘彻底)。
# 4. 支撑确认：股价回踩 MA5 或 MA10 且不破，RSI 从超买区回落到 50-60 强势中位区。
# 5. 纯正血统：排除 ST、创业板、高价股。
# ==========================================

DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'

def calculate_indicators(df):
    close = df['收盘']
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    # RSI6 
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/(loss + 1e-6)))
    return df

def screen_logic(file_path):
    try:
        # 自动识别分隔符
        with open(file_path, 'r', encoding='utf-8') as f:
            sep = '\t' if '\t' in f.readline() else ','
        df = pd.read_csv(file_path, sep=sep)
        if len(df) < 30: return None
        
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        code = str(curr['股票代码']).zfill(6)

        # --- 过滤：排除创业板和极端价格 ---
        if code.startswith('30') or curr['收盘'] < 5.0 or curr['收盘'] > 30.0: return None
        
        # --- 核心逻辑 1：拒绝追高 ---
        if curr['涨跌幅'] > 3.0: return None # 涨太多了就不看了，我们要买在启动前

        # --- 核心逻辑 2：寻找“基因” ---
        # 过去 10 天内必须有大阳线 (主力进场信号)
        recent_5 = df.iloc[-5:-1]
        has_big_sun = recent_5['涨跌幅'].max() > 6.0
        if not has_big_sun: return None
        
        # --- 核心逻辑 3：缩量回踩 ---
        # 异动那天的成交量
        max_vol_day = recent_10.loc[recent_10['涨跌幅'].idxmax(), '成交量']
        if curr['成交量'] > max_vol_day * 0.6: return None # 成交量必须大幅萎缩，代表抛压消失
        
        # --- 核心逻辑 4：趋势支撑 ---
        # 股价正在 MA5 或 MA10 附近，且没有跌破
        on_support = (curr['收盘'] >= curr['MA10'] * 0.99) and (curr['收盘'] <= curr['MA5'] * 1.02)
        
        # --- 评分 ---
        score = 70
        if on_support: score += 15
        if 50 < curr['RSI6'] < 65: score += 15 # RSI 回落到黄金中位区
        
        if score >= 85:
            return {
                '代码': code, '收盘': curr['收盘'], '涨跌幅': curr['涨跌幅'],
                '评分': score, '信号': "【潜龙出海·买点】",
                '操作建议': "该股前期有主力建仓，目前属于缩量回踩。买在阴线或平盘，止损设在MA10，博弈次日反包大阳线。"
            }
    except: return None

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    with Pool(os.cpu_count()) as p:
        results = [r for r in p.map(screen_logic, files) if r is not None]
    
    if results:
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        final_df = pd.merge(pd.DataFrame(results), names_df, left_on='代码', right_on='code', how='left')
        final_df = final_df.sort_values(by='评分', ascending=False).head(3)
        
        now = datetime.now(); folder = now.strftime('%Y%m')
        os.makedirs(folder, exist_ok=True)
        save_path = f"{folder}/dragon_strike_{now.strftime('%Y%m%d_%H%M')}.csv"
        final_df[['代码', 'name', '收盘', '涨跌幅', '评分', '信号', '操作建议']].to_csv(save_path, index=False, encoding='utf_8_sig')
