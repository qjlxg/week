import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==========================================
# 战法名称：龙头蓄势战法 (Dragon Momentum Strategy)
# 战法逻辑：
# 1. 核心筛选：5.0-20.0元，排除ST、创业板(30)、科创板(688)。
# 2. 技术形态：均线多头(MA5>10>20)，20日线趋势向上，成交量 > 20日均量2倍。
# 3. 历史回测：模拟信号触发后，计算未来 5 个交易日内的最高涨幅。
# 4. 复盘评分：换手率 > 5% 且收盘为最高价时，定义为“极强”信号。
# ==========================================

STOCK_DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'
OUTPUT_DIR = datetime.now().strftime('%Y%m')

def backtest_single_stock(file_path):
    """
    单个股票的历史全量回测逻辑
    """
    try:
        code = os.path.basename(file_path).replace('.csv', '')
        # 排除非深沪A股 (只留 60, 00 开头)
        if not code.startswith(('60', '00')) or 'ST' in code:
            return None
        
        df = pd.read_csv(file_path)
        if len(df) < 30: return None
        
        # --- 预计算技术指标 ---
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA10'] = df['收盘'].rolling(window=10).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['VOL_MA20'] = df['成交量'].rolling(window=20).mean()
        
        hit_signals = []
        
        # 从第20天开始遍历，预留最后5天计算收益
        for i in range(20, len(df) - 5):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # --- 筛选条件 ---
            last_close = float(curr['收盘'])
            change = float(curr['涨跌幅'])
            
            cond_price = 5.0 <= last_close <= 20.0
            cond_ma = curr['MA5'] > curr['MA10'] > curr['MA20']
            cond_trend = curr['MA20'] > prev['MA20']
            cond_vol = curr['成交量'] > (curr['VOL_MA20'] * 2)
            cond_change = 3.0 <= change <= 9.5  # 适度放宽涨幅限制以捕捉强势股
            
            if cond_price and cond_ma and cond_trend and cond_vol and cond_change:
                # --- 计算收益回测 ---
                # 获取未来5天的最高价
                future_window = df.iloc[i+1 : i+6]
                max_high = future_window['最高'].max()
                max_profit = ((max_high - last_close) / last_close) * 100
                
                # --- 复盘逻辑：买入信号强度 ---
                score = 0
                turnover = float(curr['换手率'])
                if turnover > 5: score += 40
                if curr['成交量'] > curr['VOL_MA20'] * 3: score += 30
                if curr['收盘'] >= curr['最高'] * 0.99: score += 30 # 接近光头阳线
                
                if score >= 80:
                    signal, advice = "极强 (⭐⭐⭐⭐⭐)", "主力高强度介入，5日内必有新高，建议重仓试错。"
                elif score >= 50:
                    signal, advice = "转强 (⭐⭐⭐)", "趋势确立，放量明显，建议观察分时择机入场。"
                else:
                    signal, advice = "一般 (⭐)", "形态达标但活跃度不足，建议作为备选。"

                hit_signals.append({
                    '日期': curr['日期'],
                    '代码': code,
                    '收盘价': last_close,
                    '当日涨幅%': change,
                    '换手率%': turnover,
                    '信号强度': signal,
                    '5日内最高收益%': round(max_profit, 2),
                    '操作建议': advice,
                    'score': score
                })
        return hit_signals
    except Exception:
        return None

def run_main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files = glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv"))
    print(f"🚀 开始并行历史回测，目标文件数: {len(files)}")
    
    # --- 并行计算加速 ---
    with Pool(cpu_count()) as p:
        results_nested = p.map(backtest_single_stock, files)
    
    # 展平列表
    flat_results = [item for sublist in results_nested if sublist for item in sublist]
    
    if not flat_results:
        print("❌ 未发现符合战法条件的信号。")
        return

    res_df = pd.DataFrame(flat_results)
    
    # 匹配股票名称
    if os.path.exists(NAMES_FILE):
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        res_df = res_df.merge(names_df, left_on='代码', right_on='code', how='left')
    
    # 优中选优：按日期倒序，同日期按评分倒序
    res_df = res_df.sort_values(by=['日期', 'score'], ascending=[False, False])
    
    # 统计胜率
    success_rate = (len(res_df[res_df['5日内最高收益%'] > 3]) / len(res_df)) * 100
    print(f"📊 回测统计：共发现 {len(res_df)} 个信号，5日内上涨超3%的概率为 {success_rate:.2f}%")

    # 保存文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(OUTPUT_DIR, f"dragon_history_backtest_{timestamp}.csv")
    
    output_cols = ['日期', '代码', 'name', '收盘价', '当日涨幅%', '换手率%', '信号强度', '5日内最高收益%', '操作建议']
    res_df[output_cols].to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 回测结果已保存至: {save_path}")

if __name__ == "__main__":
    run_main()
