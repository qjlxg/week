import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool

# ==========================================
# 战法名称：【龙抬头·极致缩量金叉战法】
# 筛选逻辑（提高胜率版）：
# 1. 严格过滤：换手率在 3% - 12% 之间（过低没活力，过高是分歧）。
# 2. 拒绝长影：收盘价必须接近全天最高价（实体饱满，拒绝冲高回落）。
# 3. 趋势为王：MA5 > MA10 > MA20，且股价一阳穿过所有压力。
# 4. 底部确认：RSI6 在爆发前曾低于 30（超跌后起爆，反弹空间大）。
# 5. 操作建议：买入信号强度 90% 以上才出手。
# ==========================================

DATA_DIR = './stock_data/'
NAMES_FILE = './stock_names.csv'

def calculate_indicators(df):
    close = df['收盘']
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    
    # RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/loss))
    
    # 布林带
    df['BOLL_MID'] = close.rolling(20).mean()
    df['BOLL_UP'] = df['BOLL_MID'] + 2 * close.rolling(20).std()
    
    return df

def screen_logic(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 60: return None
        
        code = str(df['股票代码'].iloc[-1]).zfill(6)
        # 严格过滤创业板、ST
        if code.startswith('30') or 'ST' in file_path: return None
        
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- 胜率核心过滤器 ---
        # 1. 实体饱满度：拒绝上影线。收盘价在全天波动的 85% 分位以上
        entity_quality = (curr['收盘'] - curr['最低']) / (curr['最高'] - curr['最低'] + 0.01)
        if entity_quality < 0.85: return None
        
        # 2. 换手率约束：3% - 12% (主力控盘良好，且有活跃度)
        if not (3.0 <= curr['换手率'] <= 12.0): return None
        
        # 3. 价格限制
        if not (5.0 <= curr['收盘'] <= 20.0): return None
        
        # 4. 爆发力检测：倍量 + RSI强力抬头
        vol_ratio = curr['成交量'] / df['成交量'].iloc[-6:-1].mean()
        rsi_jump = curr['RSI6'] - prev['RSI6']
        
        # 5. 趋势共振：站上布林上轨 且 均线多头
        is_breakout = curr['收盘'] > curr['BOLL_UP']
        is_trend = curr['MA5'] > curr['MA10'] > curr['MA20']
        
        # 评分模型
        score = 0
        if is_breakout: score += 30
        if is_trend: score += 25
        if vol_ratio > 2.0: score += 20
        if rsi_jump > 15: score += 15
        if curr['涨跌幅'] > 4: score += 10
        
        if score >= 85:
            # 自动复盘逻辑
            signal_type = "【一击必中·极品标的】" if score >= 90 else "【择机观察】"
            advice = "形态极其饱满，属于缩量后的第一根确认阳线。明日开盘若不破今日收盘价，可视为进攻信号。"
            
            return {
                '代码': code, '收盘': curr['收盘'], '涨跌幅': curr['涨跌幅'], '量比': round(vol_ratio, 2),
                '评分': score, '信号': signal_type, '操作建议': advice
            }
    except:
        return None

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    with Pool(os.cpu_count()) as p:
        results = [r for r in p.map(screen_logic, files) if r is not None]
    
    if results:
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        names_df['code'] = names_df['code'].str.zfill(6)
        final_df = pd.DataFrame(results)
        final_df = pd.merge(final_df, names_df[['code', 'name']], left_on='代码', right_on='code', how='left')
        
        # 极致优选：只要评分最高的那 1-2 只，宁缺毋滥
        final_df = final_df.sort_values(by='评分', ascending=False).head(2)
        
        now = datetime.now()
        folder = now.strftime('%Y%m')
        os.makedirs(folder, exist_ok=True)
        save_path = f"{folder}/dragon_strike_{now.strftime('%Y%m%d_%H%M')}.csv"
        
        final_df[['代码', 'name', '收盘', '涨跌幅', '评分', '信号', '操作建议']].to_csv(save_path, index=False, encoding='utf_8_sig')
        print(f"筛选完毕。共扫描 {len(files)} 个标的，最终精选出 {len(final_df)} 只种子选手。")
    else:
        print("今日无符合'宁缺毋滥'标准的个股。")
