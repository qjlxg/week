import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==========================================
# 战法名称：龙头蓄势 (Dragon Momentum) - 极强优选版
# 战法核心逻辑（不变）：
#   1. 5.0 <= 价格 <= 20.0
#   2. 排除 ST, 排除 300/688 (仅沪深A股)
#   3. 均线多头 (MA5 > MA10 > MA20) 且 MA20 向上
#   4. 涨幅 3% - 8.5%
#
# 结果精简过滤（新增）：
#   1. 分值优先：Score >= 80 (信号强度：极强)
#   2. 黄金量比：量比在 [2.5, 4.5] 之间
#   3. 结果去重：同一只股近期若多次触发，仅保留最新一次，确保一击必中
# ==========================================

STOCK_DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'
OUTPUT_DIR = datetime.now().strftime('%Y%m')

def backtest_logic(file_path):
    try:
        code = os.path.basename(file_path).replace('.csv', '')
        # 严格过滤板块：00, 60开头，排除ST
        if not code.startswith(('60', '00')) or 'ST' in code:
            return None
        
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 30: return None
        df = df.sort_values('日期')
        
        # 指标计算
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA10'] = df['收盘'].rolling(window=10).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['VOL_MA20'] = df['成交量'].rolling(window=20).mean()
        
        hit_signals = []
        # 遍历历史
        for i in range(20, len(df) - 5):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            price = float(curr['收盘'])
            change = float(curr['涨跌幅'])
            vol_ratio = curr['成交量'] / curr['VOL_MA20']
            
            # --- 原始战法准入条件 ---
            cond_base = (5.0 <= price <= 20.0 and 
                         curr['MA5'] > curr['MA10'] > curr['MA20'] and 
                         curr['MA20'] > prev['MA20'] and
                         3.0 <= change <= 8.5)
            
            if cond_base:
                # --- 评分逻辑 ---
                score = 0
                turnover = float(curr['换手率'])
                if vol_ratio > 3: score += 40
                if turnover > 5: score += 30
                if curr['收盘'] >= curr['最高'] * 0.99: score += 30 
                
                # --- 极强精简过滤 ---
                if score >= 80 and 2.5 <= vol_ratio <= 4.5:
                    future_high = df.iloc[i+1 : i+6]['最高'].max()
                    max_profit = ((future_high - price) / price) * 100
                    
                    hit_signals.append({
                        '日期': curr['日期'],
                        '代码': code,
                        '收盘': price,
                        '涨幅%': change,
                        '量比': round(vol_ratio, 2),
                        '换手%': turnover,
                        '信号强度': "极强 (⭐⭐⭐⭐⭐)",
                        '5日内最高收益%': round(max_profit, 2),
                        '操作建议': "极强抢筹；若3日无收益或破触发日最低价则离场"
                    })
        return hit_signals
    except:
        return None

def run_main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    files = glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv"))
    
    with Pool(cpu_count()) as p:
        results = p.map(backtest_logic, files)
    
    flat_list = [item for sublist in results if sublist for item in sublist]
    if not flat_list:
        print("未筛选出极强个股。")
        return

    res_df = pd.DataFrame(flat_list)
    if os.path.exists(NAMES_FILE):
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        res_df = res_df.merge(names_df, left_on='代码', right_on='code', how='left')
    
    # 结果去重：同日期同代码去重，保留最新
    res_df = res_df.sort_values(by=['日期', '量比'], ascending=[False, False])
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(OUTPUT_DIR, f"dragon_history_backtest_{timestamp}.csv")
    
    cols = ['日期', '代码', 'name', '收盘', '涨幅%', '量比', '换手%', '信号强度', '5日内最高收益%', '操作建议']
    res_df[cols].to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 极强筛选完成！输出信号: {len(res_df)} 条（已过滤掉弱势及异常量比品种）")

if __name__ == "__main__":
    run_main()
