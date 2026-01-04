import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 脚本名称：【风口主升浪 - 参数平衡版】
# ==============================================================================

DATA_DIR = 'stock_data'
NAMES_FILE = 'stock_names.csv'

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def process_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 250: return None
        df = df.sort_values(by='日期')
        last = df.iloc[-1]
        code = str(last['股票代码']).zfill(6)
        
        # 1. 基础过滤
        if code.startswith('30') or not (5.0 <= last['收盘'] <= 20.0): return None
        
        # 2. 筹码：连续3日换手率 5%-15%
        if not df['换手率'].tail(3).between(5, 15).all(): return None
            
        # 3. 形态：多头且在年线上方
        ma5, ma10, ma20 = df['收盘'].rolling(5).mean().iloc[-1], df['收盘'].rolling(10).mean().iloc[-1], df['收盘'].rolling(20).mean().iloc[-1]
        ma250 = df['收盘'].rolling(250).mean().iloc[-1]
        if not (ma5 > ma10 > ma20 and last['收盘'] > ma250): return None
            
        # 4. RSI 平衡区 (65-80)
        rsi = calculate_rsi(df['收盘']).iloc[-1]
        if not (65 <= rsi <= 80): return None

        # 5. 资金确认：成交量 > 1.2倍均量
        vol_ma5 = df['成交量'].rolling(5).mean().iloc[-1]
        if last['成交量'] < vol_ma5 * 1.2: return None

        return {
            "code": code, "price": last['收盘'], "turnover": last['换手率'], 
            "rsi": round(rsi, 2), "vol_ratio": round(last['成交量'] / vol_ma5, 2)
        }
    except: return None

def main():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    with Pool(cpu_count()) as p:
        results = [r for r in p.map(process_file, files) if r]
    
    if not results:
        print("建议继续观察，当前市场无高强度信号。")
        return
    
    res_df = pd.DataFrame(results)
    if os.path.exists(NAMES_FILE):
        names = pd.read_csv(NAMES_FILE); names['code'] = names['code'].astype(str).str.zfill(6)
        res_df = pd.merge(res_df, names[['code', 'name']], on='code', how='left')
    
    res_df = res_df.sort_values(by='vol_ratio', ascending=False)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
