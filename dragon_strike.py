import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool

# ==========================================
# 战法备注：【潜龙回首·KDJ共振回踩战法】
# 1. 灵敏指标：KDJ 在 60 以下发生金叉 (K上穿D)，预示短期超卖结束。
# 2. 拒绝追高：今日涨幅必须在 [-2%, 3%] 之间，买在震荡位，不买拉升位。
# 3. 强势基因：过去 10 天内必须有过一次 6% 以上的大阳线 (有主力在)。
# 4. 缩量回踩：今日成交量小于那根大阳线成交量的 60% (洗盘完成)。
# 5. 支撑确认：股价回踩 MA10 附近，且收盘价高于 MA20。
# ==========================================

DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'

def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算 KDJ 指标"""
    low_list = df['最低'].rolling(window=n).min()
    high_list = df['最高'].rolling(window=n).max()
    rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=m1-1).mean()
    df['D'] = df['K'].ewm(com=m2-1).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

def screen_logic(file_path):
    try:
        # 自动识别分隔符
        with open(file_path, 'r', encoding='utf-8') as f:
            sep = '\t' if '\t' in f.readline() else ','
        df = pd.read_csv(file_path, sep=sep)
        if len(df) < 40: return None
        
        # 计算指标
        df['MA10'] = df['收盘'].rolling(10).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        df = calculate_kdj(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        code = str(curr['股票代码']).zfill(6)

        # --- 基础过滤 ---
        if code.startswith('30') or curr['收盘'] < 5.0 or curr['收盘'] > 30.0: return None
        
        # --- 核心逻辑 1：KDJ 金叉且数值较小 ---
        # K线向上突破D线，且D线在65以下（避免高位死叉前的假金叉）
        kdj_gold_cross = (prev['K'] <= prev['D']) and (curr['K'] > curr['D'])
        if not (kdj_gold_cross and curr['D'] < 65): return None

        # --- 核心逻辑 2：严禁追高 ---
        if not (-2.0 <= curr['涨幅幅' if '涨幅幅' in curr else '涨跌幅'] <= 3.5): return None

        # --- 核心逻辑 3：寻找异动后的回踩 ---
        recent_10 = df.iloc[-10:-1]
        big_sun_day = recent_10[recent_10['涨跌幅'] > 6.0]
        if big_sun_day.empty: return None
        
        # 缩量判断：今日成交量 < 异动日成交量的 65%
        max_vol = big_sun_day['成交量'].max()
        if curr['成交量'] > max_vol * 0.65: return None
        
        # --- 趋势支撑 ---
        if curr['收盘'] < curr['MA20']: return None # 必须在20日线生命线之上

        # --- 综合评分 ---
        score = 80
        if curr['D'] < 30: score += 10 # 超低位金叉加分
        if curr['收盘'] > curr['MA10']: score += 10 # 站稳10日线加分
        
        if score >= 90:
            return {
                '代码': code, '收盘': curr['收盘'], '涨跌幅': curr['涨跌幅'],
                'K': round(curr['K'], 2), 'D': round(curr['D'], 2),
                '评分': score, '信号': "【潜龙金叉·现身】",
                '操作建议': "KDJ低位金叉共振，缩量回踩完毕。买入位置极佳，建议明日开盘介入，跌破MA20止损。"
            }
    except: return None

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    with Pool(os.cpu_count()) as p:
        results = [r for r in p.map(screen_logic, files) if r is not None]
    
    if results:
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        names_df['code'] = names_df['code'].str.zfill(6)
        final_df = pd.merge(pd.DataFrame(results), names_df[['code', 'name']], left_on='代码', right_on='code', how='left')
        
        # 优中选优
        final_df = final_df.sort_values(by=['评分', 'D'], ascending=[False, True]).head(2)
        
        now = datetime.now(); folder = now.strftime('%Y%m')
        os.makedirs(folder, exist_ok=True)
        save_path = f"{folder}/dragon_strike_{now.strftime('%Y%m%d_%H%M')}.csv"
        final_df[['代码', 'name', '收盘', '涨跌幅', 'K', 'D', '评分', '信号', '操作建议']].to_csv(save_path, index=False, encoding='utf_8_sig')
