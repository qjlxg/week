import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

"""
战法回测系统 v1.0 —— 【一种模式做一万遍】
回测逻辑：
1. 买入：满足极致起爆信号（v4.0逻辑）时，按当日收盘价买入。
2. 卖出：持有 5 个交易日后按收盘价卖出（或可自定义止盈止损）。
3. 统计：按年份计算交易次数、平均收益、胜率。
"""

def run_backtest_on_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 60: return []
        
        # 预计算必要指标
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA10'] = df['收盘'].rolling(10).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        
        trades = []
        # 遍历历史（跳过前60天，预留最后5天卖出空间）
        for i in range(60, len(df) - 5):
            curr = df.iloc[i]
            
            # --- 极致起爆逻辑筛选 ---
            # 1. 价格与板块过滤 (假设代码已在外部过滤，此处仅做价格过滤)
            if not (5.0 <= curr['收盘'] <= 20.0): continue
            
            # 2. 极致待势：前20天振幅 < 12%
            window_box = df.iloc[i-20:i]
            box_amp = (window_box['最高'].max() - window_box['最低'].min()) / window_box['最低'].min()
            if box_amp > 0.12: continue
            
            # 3. 均线多头
            if not (curr['MA5'] > curr['MA10'] > curr['MA20']): continue
            
            # 4. 爆发强度：涨幅 > 6% 且突破箱体且量比 > 3.5
            vol_ratio = curr['成交量'] / window_box['成交量'].mean()
            if curr['涨跌幅'] < 6.0 or curr['收盘'] < window_box['最高'].max() or vol_ratio < 3.5:
                continue
            
            # 5. 换手 3%-8% 且无长上影
            shadow = (curr['最高'] - curr['收盘']) / curr['收盘']
            if not (3.0 <= curr['换手率'] <= 8.0) or shadow > 0.02:
                continue
            
            # --- 命中信号，执行交易 ---
            buy_price = curr['收盘']
            sell_price = df.iloc[i+5]['收盘'] # 持有5天卖出
            profit_pct = (sell_price - buy_price) / buy_price * 100
            
            trades.append({
                'year': str(curr['日期'])[:4],
                'profit': profit_pct
            })
        return trades
    except:
        return []

def main():
    files = glob.glob('./stock_data/*.csv')
    print(f"开始回测，总计文件数: {len(files)}")

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_backtest_on_file, files))
    
    # 合并所有交易
    all_trades = [trade for sublist in results for trade in sublist]
    if not all_trades:
        print("未发现符合逻辑的交易记录")
        return

    report = pd.DataFrame(all_trades)
    
    # 按年份统计
    summary = report.groupby('year').agg(
        交易次数=('profit', 'count'),
        平均收益_pct=('profit', 'mean'),
        胜率_pct=('profit', lambda x: (x > 0).sum() / len(x) * 100)
    ).round(2)
    
    # 汇总行
    total_row = pd.DataFrame({
        '交易次数': [len(report)],
        '平均收益_pct': [report['profit'].mean()],
        '胜率_pct': [(report['profit'] > 0).sum() / len(report) * 100]
    }, index=['所有年份汇总']).round(2)
    
    final_report = pd.concat([summary, total_row])
    
    # 保存结果
    os.makedirs('backtest_results', exist_ok=True)
    save_path = f"backtest_results/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    final_report.to_csv(save_path, encoding='utf-8-sig')
    print(final_report)
    print(f"\n回测报告已保存至: {save_path}")

if __name__ == "__main__":
    main()
