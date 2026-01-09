import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==========================================
# 战法名称：龙头蓄势战法 (Dragon Momentum Strategy)
# 核心逻辑：
# 1. 价格区间：5.0 - 20.0 元（剔除低价股和超高价股）
# 2. 市场限定：仅限深沪A股，排除ST、排除30开头(创业板)、排除688(科创板)
# 3. 形态要求：
#    - 均线多头：5日 > 10日 > 20日线，且20日线向上。
#    - 量能爆发：当日成交量 > 20日平均成交量的2倍（放量突破）。
#    - 动能增强：涨跌幅在 3% - 7% 之间（非一字板，给入场机会）。
# 4. 复盘逻辑：根据换手率和量比计算“买入信号强度”。
# ==========================================

STOCK_DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'
OUTPUT_DIR = datetime.now().strftime('%Y%m')

def analyze_stock(file_path):
    try:
        # 获取股票代码
        code = os.path.basename(file_path).replace('.csv', '')
        
        # --- 过滤规则1：排除特定板块 ---
        if code.startswith(('30', '688', 'ST', '*ST')):
            return None
        
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 30:
            return None
        
        # 提取最新一天的值
        latest = df.iloc[-1]
        last_close = float(latest['收盘'])
        
        # --- 过滤规则2：价格区间 ---
        if not (5.0 <= last_close <= 20.0):
            return None

        # --- 技术分析计算 ---
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA10'] = df['收盘'].rolling(window=10).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['VOL_MA20'] = df['成交量'].rolling(window=20).mean()
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- 战法核心筛选逻辑 ---
        # 1. 均线纠缠后开始发散：5 > 10 > 20
        ma_condition = curr['MA5'] > curr['MA10'] > curr['MA20']
        # 2. 趋势向上：20日均线是增长的
        trend_condition = curr['MA20'] > prev['MA20']
        # 3. 量能突破：当日成交量 > 20日均量2倍
        volume_condition = curr['成交量'] > (curr['VOL_MA20'] * 2)
        # 4. 涨幅温和：今日涨幅在3%-8%之间，避免追高
        change_condition = 3.0 <= float(latest['涨跌幅']) <= 8.0

        if ma_condition and trend_condition and volume_condition and change_condition:
            # --- 自动复盘逻辑：评分系统 ---
            score = 0
            if curr['成交量'] > curr['VOL_MA20'] * 3: score += 40  # 巨量突破
            if float(latest['换手率']) > 5: score += 30          # 活跃度高
            if curr['收盘'] == curr['最高']: score += 30          # 光头阳线，强势
            
            # 操作建议衍生
            if score >= 80:
                advice = "【重点关注】主力强势介入，建议小仓位试错，回踩5日线加仓"
                signal = "极强 (⭐⭐⭐⭐⭐)"
            elif score >= 50:
                advice = "【先行观察】形态走好但量能尚可，观察明日前半小时表现"
                signal = "转强 (⭐⭐⭐)"
            else:
                advice = "【暂时放弃】虽然达标但爆发力不足，加入自选跟踪"
                signal = "一般 (⭐)"

            return {
                '代码': code,
                '收盘价': last_close,
                '涨跌幅': latest['涨跌幅'],
                '换手率': latest['换手率'],
                '信号强度': signal,
                '操作建议': advice,
                '分值': score
            }
    except Exception as e:
        return None
    return None

def run_strategy():
    # 创建年月目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 获取所有CSV文件
    files = glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv"))
    
    # 并行处理
    with Pool(cpu_count()) as p:
        results = p.map(analyze_stock, files)
    
    # 过滤无效结果
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("今日无符合战法条件的股票。")
        return

    # 转换为DataFrame
    res_df = pd.DataFrame(valid_results)
    
    # 匹配名称
    if os.path.exists(NAMES_FILE):
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        res_df = res_df.merge(names_df, left_on='代码', right_on='code', how='left')
        res_df = res_df.rename(columns={'name': '名称'})
    
    # 优中选优：按评分排序并只取前5名，确保“一击必中”
    res_df = res_df.sort_values(by='分值', ascending=False).head(5)
    
    # 输出结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"dragon_momentum_strategy_{timestamp}.csv"
    save_path = os.path.join(OUTPUT_DIR, filename)
    
    res_df[['代码', '名称', '收盘价', '涨跌幅', '换手率', '信号强度', '操作建议']].to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"筛选完成，结果已保存至: {save_path}")

if __name__ == "__main__":
    run_strategy()
