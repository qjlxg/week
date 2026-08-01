import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 战法名称：芙蓉出水 (精英过滤版)
# 核心逻辑：一阳穿四线 + 均线粘合 + 底部起爆
# 1. 基础过滤：5-20元，排除ST、创业板。
# 2. 突破识别：单日大阳线同时收在 5, 10, 20, 60日均线之上。
# 3. 粘合过滤：突破前均线系统高度收敛（标准差极小），确保非连续上涨后的突破。
# 4. 空间过滤：过去10个交易日累计涨幅不超过15%，防止追高接盘。
# ==========================================

def check_signal_elite(df, idx):
    """
    精英版战法逻辑判断
    """
    if idx < 60: return False
    curr = df.iloc[idx]
    
    # 1. 价格过滤 (5.0 - 20.0)
    if not (5.0 <= curr['收盘'] <= 20.0): return False
    
    # 2. 均线突破判断
    mas = [curr['ma5'], curr['ma10'], curr['ma20'], curr['ma60']]
    # 核心：收盘在四线之上，且开盘在三线(短中期)之下
    is_cross = curr['收盘'] > max(mas) and curr['开盘'] < min(mas[:3])
    if not is_cross: return False

    # 3. 动能：涨幅 > 5% (提高门槛，确保力度)
    if curr['涨跌幅'] < 5.0: return False

    # 4. 【精英过滤：均线粘合度】
    # 计算MA5, MA10, MA20在突破前的离散程度（标准差/均值）
    prev_mas = [df.iloc[idx-1]['ma5'], df.iloc[idx-1]['ma10'], df.iloc[idx-1]['ma20']]
    convergence = np.std(prev_mas) / np.mean(prev_mas)
    if convergence > 0.03: return False # 过滤掉均线太散乱的形态

    # 5. 【精英过滤：位置与空间】
    # 过去10天的累计涨幅，如果超过15%说明已经启动一段时间了，容易吃回撤
    lookback_range = df.iloc[max(0, idx-10):idx]
    recent_increase = (df.iloc[idx-1]['收盘'] - lookback_range.iloc[0]['收盘']) / lookback_range.iloc[0]['收盘']
    if recent_increase > 0.15: return False

    # 6. 【精英过滤：放量强度】
    volume_ratio = curr['成交量'] / curr['v_ma5'] if curr['v_ma5'] != 0 else 0
    if volume_ratio < 2.0: return False # 必须两倍量以上，才有真金白银

    return True

def process_single_file(file_path, name_map):
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 70: return None
        
        # 预计算指标
        df['ma5'] = df['收盘'].rolling(5).mean()
        df['ma10'] = df['收盘'].rolling(10).mean()
        df['ma20'] = df['收盘'].rolling(20).mean()
        df['ma60'] = df['收盘'].rolling(60).mean()
        df['v_ma5'] = df['成交量'].rolling(5).mean()

        code = str(df.iloc[-1]['股票代码']).zfill(6)
        stock_name = name_map.get(code, "未知")
        
        if 'ST' in stock_name or code.startswith('30'): return None

        # 历史回测采样 (持股3天)
        backtest_results = []
        for i in range(60, len(df) - 5):
            if check_signal_elite(df, i):
                buy_price = df.iloc[i+1]['开盘']
                sell_price = df.iloc[i+3]['收盘']
                backtest_results.append((sell_price - buy_price) / buy_price)

        # 今日筛选
        today_signal = None
        if check_signal_elite(df, -1):
            curr = df.iloc[-1]
            score = 80
            if curr['涨跌幅'] >= 9.8: score += 15
            if 3 <= curr['换手率'] <= 8: score += 5 # 换手率适中更稳健
            
            today_signal = {
                "代码": code, "名称": stock_name, "收盘价": curr['收盘'],
                "涨幅": f"{curr['涨跌幅']}%", "量比": round(curr['成交量']/curr['v_ma5'], 2),
                "评分": score, "操作建议": "【精英选股】均线高度粘合后的暴力突破，次日低吸为主。"
            }

        return {"backtest": backtest_results, "today": today_signal}
    except Exception:
        return None

def main():
    name_df = pd.read_csv('stock_names.csv')
    name_map = dict(zip(name_df['code'].astype(str).str.zfill(6), name_df['name']))
    files = glob.glob('stock_data/*.csv')
    
    all_profits = []
    today_candidates = []

    print(f"--- 正在运行 [精英过滤版] 并行分析 {len(files)} 个标的 ---")
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_file, f, name_map): f for f in files}
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_profits.extend(res['backtest'])
                if res['today']: today_candidates.append(res['today'])

    print("\n" + "="*45)
    print("【精英过滤版 - 历史回测报告】")
    if all_profits:
        win_rate = len([p for p in all_profits if p > 0]) / len(all_profits)
        avg_profit = np.mean(all_profits)
        print(f"有效样本: {len(all_profits)} 次 (条件更严，样本减少但质量更高)")
        print(f"历史胜率 (持股3天): {win_rate:.2%}")
        print(f"平均每单收益率: {avg_profit:.2%}")
    print("="*45 + "\n")

    if today_candidates:
        final_df = pd.DataFrame(today_candidates).sort_values(by="评分", ascending=False)
        dir_path = datetime.now().strftime("%Y-%m")
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        filename = f"{dir_path}/elite_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        final_df.to_csv(filename, index=False, encoding='utf_8_sig')
        print(f"筛选结果已保存至: {filename}")
        print(final_df)
    else:
        print("今日未筛选出符合 [精英过滤版] 条件的极致个股。")

if __name__ == "__main__":
    main()
