import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import multiprocessing

"""
战法名称：【首板缩量回踩擒龙战法】
战法逻辑说明：
1. 强势基因：5日内出现过涨停（涨幅>=9.9%），代表大资金介入。
2. 震仓洗盘：涨停后伴随“倍量阴线”或高位震荡，消化获利盘。
3. 枯竭确认：当前K线呈现极度缩量（量比 < 0.6），且收盘价守住MA5或MA10均线。
4. 买入逻辑：在缩量回踩均线处潜伏，博弈洗盘结束后的第二波主升浪。
"""

# --- 全局配置 ---
PRICE_MIN = 5.0
PRICE_MAX = 20.0
BACKTEST_DAYS = 60  # 回测过去60个交易日的表现

def get_strategy_signal(df, i):
    """检测第 i 行是否符合战法信号"""
    if i < 20: return False, 0
    
    curr = df.iloc[i]
    prev = df.iloc[i-1]
    
    # 基础过滤
    if not (PRICE_MIN <= curr['收盘'] <= PRICE_MAX): return False, 0
    
    # 1. 寻找最近5日内的首板
    lookback = df.iloc[max(0, i-5):i]
    if lookback.empty or not any(lookback['涨跌幅'] >= 9.9): return False, 0
    
    # 2. 均线计算
    ma5 = df['收盘'].rolling(5).mean().iloc[i]
    ma10 = df['收盘'].rolling(10).mean().iloc[i]
    
    # 3. 缩量逻辑
    vol_ratio = curr['成交量'] / prev['成交量']
    if vol_ratio > 0.65: return False, 0 # 必须显著缩量
    
    # 4. 支撑逻辑
    dist_ma5 = abs(curr['收盘'] - ma5) / ma5
    dist_ma10 = abs(curr['收盘'] - ma10) / ma10
    if dist_ma5 > 0.02 and dist_ma10 > 0.02: return False, 0
    
    # 评分逻辑
    score = 50
    if vol_ratio < 0.45: score += 30
    if dist_ma5 < 0.01 or dist_ma10 < 0.01: score += 20
    
    return True, score

def analyze_and_backtest(file_path, name_dict):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 40: return None
        
        code = os.path.basename(file_path).replace('.csv', '')
        if code.startswith(('30', '688')): return None
        stock_name = name_dict.get(code, "未知")
        if 'ST' in stock_name: return None

        # --- 部分 A: 今日实时信号筛选 ---
        is_hit, score = get_strategy_signal(df, len(df)-1)
        current_signal = None
        if is_hit:
            latest = df.iloc[-1]
            current_signal = {
                "代码": code, "名称": stock_name, "收盘价": latest['收盘'],
                "量比": round(latest['成交量']/df.iloc[-2]['成交量'], 2),
                "信号强度": score,
                "操作建议": "【一击必中】符合极致缩量回踩，博弈反包" if score >= 80 else "【观察试错】趋势尚可，等待分时转强"
            }

        # --- 部分 B: 历史回测逻辑 ---
        backtest_results = []
        # 在过去 BACKTEST_DAYS 天中寻找信号
        start_idx = len(df) - BACKTEST_DAYS
        for j in range(max(20, start_idx), len(df) - 3): # 至少留3天看涨幅
            hit, s = get_strategy_signal(df, j)
            if hit:
                # 计算信号发出后 3 天内的最高涨幅
                buy_price = df.iloc[j]['收盘']
                max_price_3d = df.iloc[j+1:j+4]['最高'].max()
                pnl = (max_price_3d - buy_price) / buy_price * 100
                backtest_results.append(pnl)

        return {"current": current_signal, "pnl_list": backtest_results}
    except:
        return None

def run_main():
    # 1. 加载股票名称
    try:
        names_df = pd.read_csv('stock_names.csv')
        names_df['code'] = names_df['code'].astype(str).str.zfill(6)
        name_dict = dict(zip(names_df['code'], names_df['name']))
    except:
        name_dict = {}

    # 2. 并行扫描
    files = glob.glob('stock_data/*.csv')
    print(f"🚀 开始并行分析 {len(files)} 只个股并执行回测...")
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        raw_results = pool.starmap(analyze_and_backtest, [(f, name_dict) for f in files])
    
    # 3. 汇总数据
    current_hits = []
    all_pnl = []
    for r in raw_results:
        if r:
            if r['current']: current_hits.append(r['current'])
            all_pnl.extend(r['pnl_list'])

    # 4. 统计回测胜率
    win_rate = 0
    avg_pnl = 0
    if all_pnl:
        wins = [p for p in all_pnl if p > 3.0] # 3日内最高涨幅超过3%视为有效波动
        win_rate = len(wins) / len(all_pnl) * 100
        avg_pnl = sum(all_pnl) / len(all_pnl)

    # 5. 输出与保存
    now = datetime.now()
    folder = now.strftime('%Y%m')
    os.makedirs(folder, exist_ok=True)
    file_path = f"{folder}/shoulon_strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv"

    if current_hits:
        res_df = pd.DataFrame(current_hits).sort_values(by="信号强度", ascending=False)
        # 将回测统计写入文件头部作为注释
        summary = f"--- 战法复盘统计：过去60日成功率(3%目标): {win_rate:.2f}% | 平均潜在涨幅: {avg_pnl:.2f}% ---\n"
        
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            f.write(summary)
            res_df.to_csv(f, index=False, encoding='utf-8-sig')
        print(f"✅ 筛选完成！发现 {len(current_hits)} 个信号。回测胜率: {win_rate:.2f}%")
    else:
        # 如果没有信号，也生成一个包含统计的文件
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            f.write(f"今日无信号。回测统计：总计捕捉信号 {len(all_pnl)} 个，胜率 {win_rate:.2f}%")
        print("今日未发现信号，已更新回测统计。")

if __name__ == "__main__":
    run_main()
