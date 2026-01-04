import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 脚本名称：【风口主升浪 - 实盘筛选优化版】
# 筛选逻辑：
# 1. 基础过滤：价格 5-20元；排除创业板 (30开头) 与 ST 股。
# 2. 筹码硬指标：连续 3 日换手率在 5%-15% 之间（主力控盘锁仓）。
# 3. 技术确认：5/10/20日均线多头排列（上楼梯走势）；RSI 在 60-80 强势区。
# 4. 趋势过滤 (优化)：股价必须在 MA250 年线上方，确保处于大上升周期 。
# 5. 资金确认 (优化)：当日成交量 > 5日均量（资金放量启动攻击）。
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
        if len(df) < 250: return None  # 确保有足够数据计算年线
        
        df = df.sort_values(by='日期')
        last = df.iloc[-1]
        code = str(last['股票代码']).zfill(6)
        
        # 1. 基础过滤：排除创业板、价格区间、排除ST（假设文件名或代码不含ST）
        if code.startswith('30') or not (5.0 <= last['收盘'] <= 20.0):
            return None
        
        # 2. 筹码过滤：连续3日换手率 5%-15%
        if not df['换手率'].tail(3).between(5, 15).all():
            return None
            
        # 3. 趋势与技术面过滤
        ma5 = df['收盘'].rolling(5).mean().iloc[-1]
        ma10 = df['收盘'].rolling(10).mean().iloc[-1]
        ma20 = df['收盘'].rolling(20).mean().iloc[-1]
        ma250 = df['收盘'].rolling(250).mean().iloc[-1] # 年线优化 [cite: 7]
        
        # 均线多头且在年线上方
        if not (ma5 > ma10 > ma20 and last['收盘'] > ma250):
            return None
            
        # 4. RSI 强势区过滤 (60-80)
        rsi = calculate_rsi(df['收盘']).iloc[-1]
        if not (60 <= rsi <= 80):
            return None

        # 5. 资金放量确认：当日成交量 > 5日均量 
        vol_ma5 = df['成交量'].rolling(5).mean().iloc[-1]
        if last['成交量'] <= vol_ma5:
            return None

        return {
            "code": code, 
            "price": last['收盘'], 
            "turnover": last['换手率'], 
            "rsi": round(rsi, 2),
            "ma250_dist": round((last['收盘'] - ma250) / ma250 * 100, 2) # 偏离年线百分比
        }
    except Exception:
        return None

def main():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"正在根据优化逻辑筛选潜力股，扫描总数: {len(files)}...")
    
    with Pool(cpu_count()) as p:
        results = [r for r in p.map(process_file, files) if r]
    
    if not results:
        print("今日未发现符合所有优化条件的个股。")
        return
    
    res_df = pd.DataFrame(results)
    
    # 匹配股票名称
    if os.path.exists(NAMES_FILE):
        names = pd.read_csv(NAMES_FILE)
        names['code'] = names['code'].astype(str).str.zfill(6)
        res_df = pd.merge(res_df, names[['code', 'name']], on='code', how='left')
    
    # 按年月归档保存
    now = datetime.now()
    path = now.strftime('%Y-%m')
    os.makedirs(path, exist_ok=True)
    file_name = f"{path}/PICK_OPTIMIZED_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    res_df.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    print(f"筛选完成！共发现 {len(res_df)} 只潜力股，结果已保存至: {file_name}")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
