import os
import pandas as pd
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==========================================
# 战法名称：【风口主升浪-三步筛选法】
# 核心逻辑：
# 1. 价格过滤：5元 < 收盘价 < 20元，排除ST及创业板(30开头)。
# 2. 资金逻辑：近期换手率稳定(5%-15%)，确保有主力介入而非散户乱动。
# 3. 技术形态：均线多头排列（5>10>20），确认开启“上楼梯”主升浪。
# 4. 强弱指标：RSI 处于 60-80 强势区间。
# ==========================================

DATA_DIR = 'stock_data'
NAMES_FILE = 'stock_names.csv'

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 20:
            return None
        
        # 基础数据清洗与排序
        df = df.sort_values(by='日期')
        last_row = df.iloc[-1]
        code = str(last_row['股票代码']).zfill(6)
        
        # --- 条件排除 ---
        # 1. 价格区间
        close_price = last_row['收盘']
        if not (5.0 <= close_price <= 20.0):
            return None
        
        # 2. 排除创业板(30)和ST(假设文件名或代码含ST)
        if code.startswith('30'):
            return None
        
        # --- 战法逻辑筛选 ---
        
        # A. 换手率过滤：连续3天换手率在 5%-15% 之间
        turnover_check = df['换手率'].tail(3).between(5, 15).all()
        if not turnover_check:
            return None
            
        # B. 均线多头排列：5日 > 10日 > 20日
        ma5 = df['收盘'].rolling(5).mean()
        ma10 = df['收盘'].rolling(10).mean()
        ma20 = df['收盘'].rolling(20).mean()
        
        is_bullish = ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]
        if not is_bullish:
            return None
            
        # C. RSI 指标：处于 60-80 强势区
        rsi = calculate_rsi(df['收盘'])
        current_rsi = rsi.iloc[-1]
        if not (60 <= current_rsi <= 80):
            return None

        return {"code": code, "close": close_price, "turnover": last_row['换手率'], "rsi": round(current_rsi, 2)}
    
    except Exception as e:
        return None

def main():
    # 1. 获取所有股票文件
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    # 2. 并行筛选
    with Pool(cpu_count()) as p:
        results = p.map(process_file, files)
    
    # 过滤掉 None 值
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("今日无符合战法条件的股票")
        return

    # 3. 匹配股票名称
    final_df = pd.DataFrame(valid_results)
    if os.path.exists(NAMES_FILE):
        names_df = pd.read_csv(NAMES_FILE)
        names_df['code'] = names_df['code'].astype(str).str.zfill(6)
        final_df = pd.merge(final_df, names_df[['code', 'name']], on='code', how='left')
    
    # 4. 保存结果
    now = datetime.now()
    dir_path = now.strftime('%Y-%m')
    os.makedirs(dir_path, exist_ok=True)
    
    file_name = f"pick_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    save_path = os.path.join(dir_path, file_name)
    
    final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"筛选完成，结果已保存至: {save_path}")

if __name__ == "__main__":
    main()
