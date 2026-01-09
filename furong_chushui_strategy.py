import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 战法名称：芙蓉出水 (一阳穿四线) - 并行回测版
# 战法逻辑：
# 1. 均线纠缠后，一根大阳线同时突破 5, 10, 20, 60日均线。
# 2. 限制条件：5元 < 股价 < 20元；排除ST、创业板(30开头)。
# 3. 动能：涨幅 > 4.5% 且 放量 (量比 > 1.5)。
# 4. 回测逻辑：次日开盘买入，持股3个交易日后卖出，计算成功率。
# ==========================================

def check_signal(df, idx):
    """
    战法核心逻辑判断
    """
    if idx < 60: return False
    curr = df.iloc[idx]
    
    # 1. 价格过滤
    if not (5.0 <= curr['收盘'] <= 20.0): return False
    
    # 2. 均线突破判断 (计算 MA5, 10, 20, 60)
    # 注意：此处假设传入的 df 已经计算好了均线
    mas = [curr['ma5'], curr['ma10'], curr['ma20'], curr['ma60']]
    
    # 收盘价高于所有均线，且开盘价低于至少三条短中期均线（穿透性）
    is_cross = curr['收盘'] > max(mas) and curr['开盘'] < min(mas[:3])
    
    # 3. 涨幅与量能
    is_strong = curr['涨跌幅'] > 4.5
    volume_ratio = curr['成交量'] / curr['v_ma5'] if curr['v_ma5'] != 0 else 0
    
    return is_cross and is_strong and volume_ratio > 1.5

def process_single_file(file_path, name_map):
    """
    单个文件的处理函数：同时完成今日筛选和历史回测采样
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 70: return None
        
        # 预计算所有技术指标
        df['ma5'] = df['收盘'].rolling(5).mean()
        df['ma10'] = df['收盘'].rolling(10).mean()
        df['ma20'] = df['收盘'].rolling(20).mean()
        df['ma60'] = df['收盘'].rolling(60).mean()
        df['v_ma5'] = df['成交量'].rolling(5).mean()

        code = str(df.iloc[-1]['股票代码']).zfill(6)
        stock_name = name_map.get(code, "未知")
        
        # 基础过滤
        if 'ST' in stock_name or code.startswith('30'): return None

        # --- 部分A：历史回测数据采样 ---
        backtest_results = []
        # 遍历历史数据（留出最后5天不作为信号点，用于验证卖出）
        for i in range(60, len(df) - 5):
            if check_signal(df, i):
                buy_price = df.iloc[i+1]['开盘']  # 次日开盘买
                sell_price = df.iloc[i+3]['收盘'] # 持股3日卖
                profit = (sell_price - buy_price) / buy_price
                backtest_results.append(profit)

        # --- 部分B：今日信号筛选 ---
        today_signal = None
        if check_signal(df, -1):
            curr = df.iloc[-1]
            # 评分系统
            score = 70
            if curr['涨跌幅'] >= 9.8: score += 20
            if 3 <= curr['换手率'] <= 10: score += 10
            
            advice = "【极强】建议次日择机介入" if score >= 90 else "【走强】建议小仓位试错"
            today_signal = {
                "代码": code, "名称": stock_name, "收盘价": curr['收盘'],
                "涨跌幅": f"{curr['涨跌幅']}%", "换手率": f"{curr['换手率']}%",
                "量比": round(curr['成交量']/curr['v_ma5'], 2),
                "评分": score, "操作建议": advice
            }

        return {"backtest": backtest_results, "today": today_signal}
    except Exception:
        return None

def main():
    # 1. 初始化
    name_df = pd.read_csv('stock_names.csv')
    name_map = dict(zip(name_df['code'].astype(str).str.zfill(6), name_df['name']))
    files = glob.glob('stock_data/*.csv')
    
    all_backtest_profits = []
    today_candidates = []

    print(f"--- 正在并行分析 {len(files)} 个标的的历史表现与今日信号 ---")
    
    # 2. 并行执行
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, f, name_map): f for f in files}
        for future in as_completed(future_to_file):
            res = future.result()
            if res:
                if res['backtest']:
                    all_backtest_profits.extend(res['backtest'])
                if res['today']:
                    today_candidates.append(res['today'])

    # 3. 输出回测统计
    print("\n" + "="*40)
    print("【历史回测报告 - 芙蓉出水战法】")
    if all_backtest_profits:
        win_rate = len([p for p in all_backtest_profits if p > 0]) / len(all_backtest_profits)
        avg_profit = np.mean(all_backtest_profits)
        print(f"回测样本总数: {len(all_backtest_profits)} 次")
        print(f"历史胜率 (持股3天): {win_rate:.2%}")
        print(f"平均每单收益率: {avg_profit:.2%}")
        if avg_profit < 0:
            print("⚠️ 警示：当前市场环境下该战法平均收益为负，请严格执行止损。")
    else:
        print("未获取到足够的历史回测样本。")
    print("="*40 + "\n")

    # 4. 输出今日筛选结果
    if today_candidates:
        final_df = pd.DataFrame(today_candidates).sort_values(by="评分", ascending=False)
        
        now = datetime.now()
        dir_path = now.strftime("%Y-%m")
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        
        filename = f"{dir_path}/furong_chushui_strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        final_df.to_csv(filename, index=False, encoding='utf_8_sig')
        
        print(f"今日复盘完成！选出 {len(final_df)} 只个股。")
        print(f"结果已保存至: {filename}")
        # 在控制台打印前5名方便快速查看
        print(final_df[['代码', '名称', '评分', '操作建议']].head())
    else:
        print("今日未筛选出符合战法的个股。")

if __name__ == "__main__":
    main()
