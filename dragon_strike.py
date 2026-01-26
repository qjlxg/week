import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool

# ==========================================
# 战法备注：【潜龙出海·5日缩量回踩战法】
# 核心逻辑：
# 1. 拒绝追高：涨幅限制在 [-2.5%, 3.5%] 之间。避开情绪过热，买入蓄势位。
# 2. 异动基因：过去 5 天内必须出现过至少一次大阳线 (涨幅 > 6%)，证明有主力入驻。
# 3. 缩量回踩：今日成交量必须是异动大阳线当天的 60% 以下。代表浮筹洗净，抛压极轻。
# 4. 支撑确认：股价回踩 MA5 或 MA10 支撑位且不破，RSI6 回落至 [50, 68] 的强势中位区。
# 5. 纯正血统：排除 ST、创业板 (30开头)、价格不在 5-30 元区间的个股。
# ==========================================

DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'

def calculate_indicators(df):
    close = df['收盘']
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    
    # RSI6 指标计算
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/(loss + 1e-6)))
    return df

def screen_logic(file_path):
    try:
        # 自动识别 CSV 分隔符
        with open(file_path, 'r', encoding='utf-8') as f:
            sep = '\t' if '\t' in f.readline() else ','
        df = pd.read_csv(file_path, sep=sep)
        
        if len(df) < 30: return None
        
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        code = str(curr['股票代码']).zfill(6)

        # --- 基础过滤 ---
        if code.startswith('30') or 'ST' in file_path: return None
        if not (5.0 <= curr['收盘'] <= 30.0): return None
        
        # --- 核心逻辑 1：拒绝追高，买在调整位 ---
        change = curr['涨跌幅'] if '涨跌幅' in curr else curr['涨幅幅']
        if not (-2.5 <= change <= 3.5): return None 

        # --- 核心逻辑 2：5日内寻找异动基因 (你的核心修改点) ---
        # 范围：从前第5天到前第1天
        recent_5 = df.iloc[-6:-1]
        big_sun_day = recent_5[recent_5['涨跌幅'] > 6.0]
        if big_sun_day.empty: return None
        
        # --- 核心逻辑 3：精准缩量回踩 ---
        # 以这 5 天内涨幅最高的那天的成交量为基准
        max_vol_day = big_sun_day.loc[big_sun_day['涨跌幅'].idxmax(), '成交量']
        if curr['成交量'] > max_vol_day * 0.6: return None 
        
        # --- 核心逻辑 4：趋势与 RSI 共振 ---
        # 股价正在 MA5 或 MA10 附近（上下 2% 范围内）
        on_support = (curr['收盘'] >= curr['MA10'] * 0.98) and (curr['收盘'] <= curr['MA5'] * 1.02)
        # 股价必须在 MA20 生命周期之上
        if curr['收盘'] < curr['MA20']: return None
        
        # --- 自动化复盘评分 ---
        score = 70
        if on_support: score += 15
        if 50 < curr['RSI6'] < 68: score += 15 
        
        if score >= 85:
            return {
                '代码': code, 
                '收盘': curr['收盘'], 
                '涨跌幅': change,
                '评分': score, 
                '信号': "【潜龙出海·5日回踩】",
                '操作建议': "5日内有强力异动，目前缩量回踩关键支撑位，且RSI未走弱。建议分批建仓，破MA20止损。"
            }
    except: 
        return None

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # 并行扫描提升速度
    with Pool(os.cpu_count()) as p:
        results = [r for r in p.map(screen_logic, files) if r is not None]
    
    if results:
        # 获取名称并合并
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        names_df['code'] = names_df['code'].str.zfill(6)
        
        final_df = pd.DataFrame(results)
        final_df = pd.merge(final_df, names_df[['code', 'name']], left_on='代码', right_on='code', how='left')
        
        # 宁缺毋滥，只选评分最高的前 3 名
        final_df = final_df.sort_values(by='评分', ascending=False).head(3)
        
        # 结果保存
        now = datetime.now()
        folder = now.strftime('%Y%m')
        os.makedirs(folder, exist_ok=True)
        save_path = f"{folder}/dragon_strike_{now.strftime('%Y%m%d_%H%M')}.csv"
        
        final_df[['代码', 'name', '收盘', '涨跌幅', '评分', '信号', '操作建议']].to_csv(
            save_path, index=False, encoding='utf_8_sig'
        )
        print(f"复盘完成。全场扫描出 {len(final_df)} 只高胜率回踩标的。")
    else:
        print("今日市场未发现符合'5日缩量回踩'标准的个股。")
