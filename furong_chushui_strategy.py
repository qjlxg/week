import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 战法名称：芙蓉出水 (一阳穿四线) - 增强回测版
# 核心逻辑：
# 1. 价格区间：5元 - 20元，排除ST和创业板。
# 2. 均线突破：单日大阳线同时收在 5, 10, 20, 60日均线之上。
# 3. 动能要求：涨幅 > 4.5%，成交量明显放大（量比 > 1.5）。
# 4. 买卖逻辑：回测模拟信号出现次日以开盘价买入，持股 N 天后卖出。
# ==========================================

def analyze_stock(file_path, name_map, backtest_mode=False):
    """
    分析单个股票：支持当日选股和历史回测两种模式
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 70: return None
        
        # 统一计算均线
        df['ma5'] = df['收盘'].rolling(5).mean()
        df['ma10'] = df['收盘'].rolling(10).mean()
        df['ma20'] = df['收盘'].rolling(20).mean()
        df['ma60'] = df['收盘'].rolling(60).mean()
        df['v_ma5'] = df['成交量'].rolling(5).mean()

        code = str(df.iloc[-1]['股票代码']).zfill(6)
        # 基础过滤：价格、ST、创业板
        if 'ST' in name_map.get(code, '') or code.startswith('30'): return None

        def check_signal(data_slice, idx):
            curr = data_slice.iloc[idx]
            # 价格过滤
            if not (5.0 <= curr['收盘'] <= 20.0): return False
            # 均线位置判断
            mas = [curr['ma5'], curr['ma10'], curr['ma20'], curr['ma60']]
            is_cross = curr['收盘'] > max(mas) and curr['开盘'] < min(mas[:3])
            # 动能判断
            is_strong = curr['涨跌幅'] > 4.5
            volume_ratio = curr['成交量'] / curr['v_ma5'] if curr['v_ma5'] != 0 else 0
            return is_cross and is_strong and volume_ratio > 1.5

        if not backtest_mode:
            # --- 模式1：今日实时复盘 ---
            if not check_signal(df, -1): return None
            curr = df.iloc[-1]
            score = 60
            if curr['涨跌幅'] >= 9.8: score += 20
            if 3 <= curr['换手率'] <= 10: score += 10
            if curr['收盘'] > curr['ma60']: score += 10
            
            advice = "【极强】重仓关注，次日观察支撑" if score >= 90 else "【走强】建议底仓试错"
            return {
                "代码": code, "名称": name_map.get(code, "未知"),
                "收盘价": curr['收盘'], "涨跌幅": curr['涨跌幅'],
                "量比": round(curr['成交量']/curr['v_ma5'], 2), "评分": score, "操作建议": advice
            }
        else:
            # --- 模式2：历史回测 ---
            hits = []
            # 遍历历史（留出5天看后市表现）
            for i in range(60, len(df) - 5):
                if check_signal(df, i):
                    buy_price = df.iloc[i+1]['开盘'] # 次日开盘买入
                    sell_price_3d = df.iloc[i+3]['收盘'] # 持股3天卖出
                    profit = (sell_price_3d - buy_price) / buy_price
                    hits.append(profit)
            return hits

    except Exception:
        return None

def run_backtest(files, name_map):
    """ 运行回测逻辑 """
    print("\n--- 正在启动历史回测 (模拟持股3天) ---")
    all_profits = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_stock, f, name_map, True) for f in files]
        for f in futures:
            res = f.result()
            if res: all_profits.extend(res)
    
    if all_profits:
        win_rate = len([p for p in all_profits if p > 0]) / len(all_profits)
        avg_profit = np.mean(all_profits)
        print(f"回测样本数: {len(all_profits)}")
        print(f"历史胜率: {win_rate:.2%}")
        print(f"平均每单收益: {avg_profit:.2%}")
        print("------------------------------------\n")

def main():
    # 1. 加载映射
    name_df = pd.read_csv('stock_names.csv')
    name_map = dict(zip(name_df['code'].astype(str).str.zfill(6), name_df['name']))
    files = glob.glob('stock_data/*.csv')
    
    # 2. 先跑回测，看战法历史表现
    run_backtest(files, name_map)
    
    # 3. 执行今日筛选
    print("--- 正在执行今日复盘筛选 ---")
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_stock, f, name_map, False) for f in files]
        for f in futures:
            res = f.result()
            if res: results.append(res)
            
    if results:
        final_df = pd.DataFrame(results).sort_values(by="评分", ascending=False)
        now = datetime.now()
        dir_path = now.strftime("%Y-%m")
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        filename = f"{dir_path}/furong_chushui_strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        final_df.to_csv(filename, index=False, encoding='utf_8_sig')
        print(f"今日筛选完成！结果已存至: {filename}")
        print(final_df[['代码', '名称', '评分', '操作建议']])
    else:
        print("今日未发现信号。")

if __name__ == "__main__":
    main()
