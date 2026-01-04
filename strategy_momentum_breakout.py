import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 脚本名称：【风口主升浪 - 精选筛选版】
# 逻辑说明：
# 1. 基础过滤：价格 5-20元；排除创业板 (30开头) [cite: 1]。
# 2. 筹码锁仓：连续 3 日换手率在 5%-15% [cite: 3, 7]。
# 3. 极强技术：5/10/20日均线多头排列；RSI 提升至 70-80 极强区 。
# 4. 趋势保护：股价在 MA250 年线上方，且乖离率不超过 100% [cite: 7, 9]。
# 5. 爆发确认：成交量 > 5日均量的 1.5 倍（倍量启动）。
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
        
        # 1. 基础过滤：代码、价格
        if code.startswith('30') or not (5.0 <= last['收盘'] <= 20.0):
            return None
        
        # 2. 筹码硬指标：连续3日换手率 5%-15% [cite: 3]
        if not df['换手率'].tail(3).between(5, 15).all():
            return None
            
        # 3. 均线形态：多头且在年线上方 [cite: 4, 7]
        ma5 = df['收盘'].rolling(5).mean().iloc[-1]
        ma10 = df['收盘'].rolling(10).mean().iloc[-1]
        ma20 = df['收盘'].rolling(20).mean().iloc[-1]
        ma250 = df['收盘'].rolling(250).mean().iloc[-1]
        if not (ma5 > ma10 > ma20 and last['收盘'] > ma250):
            return None
            
        # 4. 严苛过滤：RSI 必须在 70-80 (极强区) 
        rsi = calculate_rsi(df['收盘']).iloc[-1]
        if not (70 <= rsi <= 80):
            return None

        # 5. 资金确认：成交量倍量启动 (>= 1.5倍)
        vol_ma5 = df['成交量'].rolling(5).mean().iloc[-1]
        if last['成交量'] < vol_ma5 * 1.5:
            return None

        # 6. 避坑逻辑：股价离年线太远(超过100%)的不看 
        ma250_dist = (last['收盘'] - ma250) / ma250
        if ma250_dist > 1.0:
            return None

        return {
            "code": code, 
            "price": last['收盘'], 
            "turnover": last['换手率'], 
            "rsi": round(rsi, 2),
            "vol_ratio": round(last['成交量'] / vol_ma5, 2), # 成交量倍数
            "ma250_dist": round(ma250_dist * 100, 2)
        }
    except Exception:
        return None

def main():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"[{datetime.now()}] 正在启动极强版筛选，扫描文件: {len(files)}...")
    
    with Pool(cpu_count()) as p:
        results = [r for r in p.map(process_file, files) if r]
    
    if not results:
        print("今日严苛条件下未发现符合条件的个股，建议空仓休息。")
        return
    
    res_df = pd.DataFrame(results)
    
    # 匹配名称
    if os.path.exists(NAMES_FILE):
        names = pd.read_csv(NAMES_FILE)
        names['code'] = names['code'].astype(str).str.zfill(6)
        res_df = pd.merge(res_df, names[['code', 'name']], on='code', how='left')
    
    # 结果排序：优先展示成交量比最高（最强攻击）的
    res_df = res_df.sort_values(by='vol_ratio', ascending=False)
    
    # 保存结果
    now = datetime.now()
    path = now.strftime('%Y-%m')
    os.makedirs(path, exist_ok=True)
    file_name = f"{path}/DAILY_PICK_HIGH_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    res_df.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    print(f"筛选完成！精选出 {len(res_df)} 只高爆发潜力股，详情如下：")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
