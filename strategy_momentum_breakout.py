import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 脚本名称：【风口主升浪 - 安全进场版】
# 核心改动：
# 1. RSI 限制在 60-70 之间，规避超买风险。
# 2. 增加 5日均线乖离率控制 (股价/MA5 <= 1.05)，防止追高。
# 3. 保持年线过滤与量比确认，确保趋势与资金支持。
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
            
        # 3. 趋势形态：均线多头且在年线上方
        ma5 = df['收盘'].rolling(5).mean().iloc[-1]
        ma10 = df['收盘'].rolling(10).mean().iloc[-1]
        ma20 = df['收盘'].rolling(20).mean().iloc[-1]
        ma250 = df['收盘'].rolling(250).mean().iloc[-1]
        if not (ma5 > ma10 > ma20 and last['收盘'] > ma250): return None
            
        # 4. 【核心优化】RSI 限制在 60-70，避开 75+ 的超买区
        rsi = calculate_rsi(df['收盘']).iloc[-1]
        if not (60 <= rsi <= 70): return None

        # 5. 【增加风控】乖离率控制：股价距离5日线不能超过 5%，防止急涨后回踩
        if last['收盘'] > ma5 * 1.05: return None

        # 6. 资金确认：成交量温和放大 (1.2 - 2.0 倍之间，避免放巨量见顶)
        vol_ma5 = df['成交量'].rolling(5).mean().iloc[-1]
        vol_ratio = last['成交量'] / vol_ma5
        if not (1.2 <= vol_ratio <= 2.5): return None

        return {
            "code": code, "price": last['收盘'], "turnover": last['换手率'], 
            "rsi": round(rsi, 2), "vol_ratio": round(vol_ratio, 2),
            "dist_ma5": round((last['收盘']-ma5)/ma5*100, 2) # 距5日线比例
        }
    except: return None

def main():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"[{datetime.now()}] 启动安全版筛选（规避高RSI与乖离）...")
    with Pool(cpu_count()) as p:
        results = [r for r in p.map(process_file, files) if r]
    
    if not results:
        print("当前市场无“安全区”强势股，建议继续等待。")
        return
    
    res_df = pd.DataFrame(results)
    if os.path.exists(NAMES_FILE):
        names = pd.read_csv(NAMES_FILE); names['code'] = names['code'].astype(str).str.zfill(6)
        res_df = pd.merge(res_df, names[['code', 'name']], on='code', how='left')
    
    # 按量比排序
    res_df = res_df.sort_values(by='vol_ratio', ascending=False)
    
    # 结果归档
    now = datetime.now()
    file_name = f"SAFE_PICK_{now.strftime('%Y%m%d')}.csv"
    res_df.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    print(f"筛选完成！找到 {len(res_df)} 只处于安全启动区的潜力股：")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
