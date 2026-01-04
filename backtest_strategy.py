import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

"""
战法名称：【一种模式做一万遍】—— 极速回测系统 v4.2 (参数平衡版)
核心逻辑微调：
1. 振幅放宽：20日箱体振幅由 12% 放宽至 18% (适应 A 股波动)。
2. 量比调整：量比由 3.5 倍调整为 2.0 倍 (3.5倍属于极端放量)。
3. 换手放宽：换手率上限由 8% 放宽至 12% (容纳更多主力行为)。
"""

def process_single_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 60: return None
        
        # 提取股票代码并过滤
        code = str(df['股票代码'].iloc[0]).zfill(6)
        if code.startswith(('30', '688', '8', '4')): return None
        if not (code.startswith('60') or code.startswith('00')): return None

        # --- 向量化指标计算 ---
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA10'] = df['收盘'].rolling(10).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        
        # 过去20天箱体指标 (不含当天)
        df['box_high'] = df['最高'].rolling(20).shift(1).max()
        df['box_low'] = df['最低'].rolling(20).shift(1).min()
        df['box_amp'] = (df['box_high'] - df['box_low']) / df['box_low']
        df['avg_vol_20'] = df['成交量'].rolling(20).shift(1).mean()
        
        # 未来5天收益率
        df['future_return'] = (df['收盘'].shift(-5) - df['收盘']) / df['收盘'] * 100

        # --- 战法条件判定 (v4.2 优化版) ---
        cond = (
            (df['收盘'] >= 5.0) & (df['收盘'] <= 25.0) &                # 价格稍微放宽
            (df['box_amp'] <= 0.18) &                                 # 振幅限制(18%)
            (df['MA5'] > df['MA20']) &                                # 均线多头(简化)
            (df['涨跌幅'] >= 4.0) &                                    # 涨幅强度(4%以上)
            (df['收盘'] > df['box_high']) &                            # 突破箱体
            (df['成交量'] > df['avg_vol_20'] * 2.0) &                  # 量比(2.0倍)
            (df['换手率'] >= 2.0) & (df['换手率'] <= 12.0) &            # 换手(2-12%)
            ((df['最高'] - df['收盘']) / df['收盘'] <= 0.03)           # 影线限制
        )
        
        trades = df[cond][['日期', 'future_return']].dropna()
        if trades.empty: return None
        
        trades['year'] = trades['日期'].astype(str).str[:4]
        return trades[['year', 'future_return']]
    except:
        return None

def main():
    stock_data_path = './stock_data/*.csv'
    files = glob.glob(stock_data_path)
    print(f"🚀 启动并行回测引擎 v4.2... 目标文件数: {len(files)}")

    with ProcessPoolExecutor() as executor:
        res_list = list(executor.map(process_single_file, files))
    
    valid_dfs = [r for r in res_list if r is not None]
    if not valid_dfs:
        print("❌ 条件依然过严，未发现交易记录。请检查 csv 中的'日期'列格式是否为 YYYYMMDD 或 YYYY-MM-DD。")
        return

    all_trades = pd.concat(valid_dfs)
    
    # 统计报表
    summary = all_trades.groupby('year')['future_return'].agg(
        交易次数='count',
        平均收益_pct='mean',
        胜率_pct=lambda x: (x > 0).sum() / len(x) * 100
    ).round(2)
    
    total_row = pd.DataFrame({
        '交易次数': [len(all_trades)],
        '平均收益_pct': [all_trades['future_return'].mean()],
        '胜率_pct': [(all_trades['future_return'] > 0).sum() / len(all_trades) * 100]
    }, index=['所有年份汇总']).round(2)
    
    final_report = pd.concat([summary, total_row])
    
    os.makedirs('backtest_results', exist_ok=True)
    report_path = f"backtest_results/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    final_report.to_csv(report_path, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(final_report)
    print("="*40)
    print(f"✅ 成功！报告已保存至: {report_path}")

if __name__ == "__main__":
    main()
