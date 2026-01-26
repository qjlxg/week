import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool

# ==========================================
# 战法名称：【云起龙骧·共振强攻战法】
# 逻辑说明：
# 1. 价格区间：5.0 - 20.0元，严格过滤非深沪A股（排除30, 688, ST）。
# 2. 技术共振：
#    - 均线：收盘价站上 5/10/20/60日均线，且均线呈发散状。
#    - 布林带：股价突破上轨（BOLL_UP），预示进入单边拉升。
#    - 动能：RSI6 > 80（进入极强区）；MACD DIF > DEA 且红柱增长。
#    - 量能：成交量相比前5日均值放大2倍以上（倍量起爆）。
# ==========================================

DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'

def calculate_indicators(df):
    """手写技术指标，摆脱对TA-Lib的依赖，速度极快"""
    close = df['收盘']
    
    # 1. 移动平均线
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    
    # 2. RSI6 (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/loss))
    
    # 3. BOLL (20, 2)
    df['BOLL_MID'] = close.rolling(20).mean()
    df['BOLL_UP'] = df['BOLL_MID'] + 2 * close.rolling(20).std()
    
    # 4. MACD (12, 26, 9)
    df['EMA12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA26'] = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = (df['DIF'] - df['DEA']) * 2
    
    return df

def screen_logic(file_path):
    try:
        # 强制指定列格式，防止读取错误
        df = pd.read_csv(file_path, sep='\t') if '\t' in open(file_path).read() else pd.read_csv(file_path)
        if len(df) < 60: return None
        
        # 提取最新行
        code = str(df['股票代码'].iloc[-1]).zfill(6)
        
        # --- 严格过滤条件 ---
        # 1. 价格区间
        last_price = df['收盘'].iloc[-1]
        if not (5.0 <= last_price <= 20.0): return None
        
        # 2. 排除30开头(创业板)、排除ST (基于文件名或代码判断)
        if code.startswith('30'): return None
        
        # 3. 指标计算
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- 战法核心判断 ---
        # A. 倍量确认 (当前成交量 > 5日均量 * 2)
        vol_avg5 = df['成交量'].iloc[-6:-1].mean()
        vol_ratio = curr['成交量'] / vol_avg5 if vol_avg5 > 0 else 0
        
        # B. 均线与布林突破 (一阳穿多线 + 站上上轨)
        is_breakout = (curr['收盘'] > curr['MA5']) and (curr['收盘'] > curr['BOLL_UP'])
        
        # C. 指标共振 (RSI极强 + MACD向上)
        is_strong = (curr['RSI6'] > 75) and (curr['MACD_HIST'] > prev['MACD_HIST'])
        
        # --- 评分系统 (优中选优) ---
        score = 0
        if is_breakout: score += 40
        if vol_ratio >= 2.0: score += 30
        if is_strong: score += 20
        if curr['涨跌幅'] > 7: score += 10 # 强势上涨加分
        
        if score >= 80:
            # 信号定义
            signal = "【一击必中】" if score >= 90 else "【观察博弈】"
            advice = "核心标的：放量突破禁锢，建议明日集合竞价观察，若高开在1%-3%可果断介入。" if score >= 90 \
                     else "试错标的：指标已走强但量能尚可，建议轻仓试盘，跌破MA5止损。"
            
            return {
                '代码': code, '收盘': last_price, '涨跌幅': curr['涨跌幅'],
                '量比': round(vol_ratio, 2), 'RSI6': round(curr['RSI6'], 2),
                '评分': score, '信号': signal, '操作建议': advice
            }
            
    except Exception as e:
        return None

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # 利用多进程加快扫描 stock_data 目录
    with Pool(os.cpu_count()) as p:
        results = [r for r in p.map(screen_logic, files) if r is not None]
    
    if results:
        # 匹配股票名称
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        names_df['code'] = names_df['code'].str.zfill(6)
        
        final_df = pd.DataFrame(results)
        final_df = pd.merge(final_df, names_df[['code', 'name']], left_on='代码', right_on='code', how='left')
        
        # 过滤空值并按评分排序
        final_df = final_df.sort_values(by='评分', ascending=False).head(3) # 极致优选，只要前3名
        
        # 保存到年月目录
        now = datetime.now()
        folder = now.strftime('%Y%m')
        os.makedirs(folder, exist_ok=True)
        save_name = f"dragon_strike_{now.strftime('%Y%m%d_%H%M')}.csv"
        
        # 输出最终结果
        final_cols = ['代码', 'name', '收盘', '涨跌幅', '量比', 'RSI6', '评分', '信号', '操作建议']
        final_df[final_cols].to_csv(f"{folder}/{save_name}", index=False, encoding='utf_8_sig')
        print(f"复盘完成！发现潜力标的 {len(final_df)} 只。")
    else:
        print("今日未发现符合共振战法标的，建议持币待仓。")
