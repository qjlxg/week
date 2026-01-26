import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool

# ==========================================
# 战法名称：【云起龙骧·共振强攻战法】
# 买入逻辑：
# 1. 价格区间：5元 - 20元，排除ST和创业板。
# 2. 趋势共振：股价一阳穿多线（5/10/20/60 MA）。
# 3. 动能爆发：当日成交量 > 前5日平均成交量2倍（量比>2）。
# 4. 指标共振：RSI(6) 突破 75；布林带开始向上开口；MACD红柱增长。
# 操作建议：优中选优，仅输出综合评分最高的标的。
# ==========================================

DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'

def calculate_indicators(df):
    """计算核心技术指标"""
    # 均线
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA10'] = df['收盘'].rolling(10).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['MA60'] = df['收盘'].rolling(60).mean()
    
    # RSI (简易计算)
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss
    df['RSI6'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['BOLL_MID'] = df['收盘'].rolling(20).mean()
    df['BOLL_UP'] = df['BOLL_MID'] + 2 * df['收盘'].rolling(20).std()
    
    # MACD (简易)
    df['EMA12'] = df['收盘'].ewm(span=12).mean()
    df['EMA26'] = df['收盘'].ewm(span=26).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    return df

def screen_logic(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if len(df) < 60: return None
        
        # 基础过滤：排除创业板(30)和价格区间
        code = str(df['股票代码'].iloc[-1]).zfill(6)
        last_close = df['收盘'].iloc[-1]
        
        if code.startswith('30') or last_close < 5.0 or last_close > 20.0:
            return None
        
        # 计算指标
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. 量能条件：倍量起爆
        vol_ratio = curr['成交量'] / df['成交量'].iloc[-6:-1].mean()
        
        # 2. 图形条件：一阳穿多线且站上布林上轨
        one_pierce_all = (curr['收盘'] > curr['MA5']) and (curr['收盘'] > curr['MA20']) and (curr['收盘'] > curr['MA60'])
        boll_break = curr['收盘'] > curr['BOLL_UP']
        
        # 3. 指标条件：RSI强势且MACD红柱放量
        rsi_strong = curr['RSI6'] > 75
        macd_grow = curr['MACD'] > prev['MACD'] and curr['MACD'] > 0
        
        # 综合评分逻辑 (满分100)
        score = 0
        if one_pierce_all: score += 30
        if boll_break: score += 20
        if rsi_strong: score += 20
        if vol_ratio > 2: score += 20
        if macd_grow: score += 10
        
        if score >= 80: # 只选高分种子
            signal = "强烈买入" if score >= 90 else "轻仓试错"
            advice = "形态完美，量价齐升，建议明日回踩MA5不破加仓" if score >= 90 else "动能充足但需防范冲高回落"
            
            return {
                '代码': code,
                '收盘': curr['收盘'],
                '涨跌幅': curr['涨跌幅'],
                '成交量': curr['成交量'],
                '量比': round(vol_ratio, 2),
                'RSI6': round(curr['RSI6'], 2),
                '强度评分': score,
                '信号': signal,
                '操作建议': advice
            }
    except Exception:
        return None

if __name__ == "__main__":
    # 1. 扫描文件并并行处理
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    with Pool(os.cpu_count()) as p:
        results = p.map(screen_logic, files)
    
    results = [r for r in results if r is not None]
    
    # 2. 匹配名称
    if results:
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        final_df = pd.DataFrame(results)
        names_df['code'] = names_df['code'].str.zfill(6)
        final_df = pd.merge(final_df, names_df, left_on='代码', right_on='code', how='left')
        
        # 优中选优：按评分降序，只取前5名
        final_df = final_df.sort_values(by='强度评分', ascending=False).head(5)
        
        # 3. 保存结果
        now = datetime.now()
        dir_path = now.strftime('%Y%m')
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        
        file_name = f"dragon_strike_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        save_path = os.path.join(dir_path, file_name)
        
        # 整理列顺序
        cols = ['代码', 'name', '收盘', '涨跌幅', '量比', 'RSI6', '强度评分', '信号', '操作建议']
        final_df[cols].to_csv(save_path, index=False, encoding='utf_8_sig')
        print(f"成功筛选出 {len(final_df)} 只潜力股，结果已存至 {save_path}")
    else:
        print("今日无符合【云起龙骧】战法标的，保持空仓观望。")
