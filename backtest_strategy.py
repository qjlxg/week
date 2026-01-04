import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

"""
战法名称：【一种模式做一万遍】—— 极速回测系统 v4.1
回测逻辑：
- 筛选：5-20元, 非ST/创业板/科创板/北交所。
- 形态：20日振幅<12%, 均线多头, 涨幅>6%, 突破箱体, 量比>3.5, 换手3-8%, 无长上影。
- 卖出：买入后第 5 个交易日按收盘价卖出。
"""

def process_single_file(file_path):
    """
    单个文件向量化处理函数，极大地提升单核运行效率
    """
    try:
        df = pd.read_csv(file_path)
        if len(df) < 60: return None
        
        # 提取股票代码（假设文件名包含代码或从列中提取）
        code = str(df['股票代码'].iloc[0]).zfill(6)
        # 过滤创业板/科创板/北交所
        if code.startswith(('30', '688', '8', '4')): return None
        if not (code.startswith('60') or code.startswith('00')): return None

        # --- 向量化指标计算 ---
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA10'] = df['收盘'].rolling(10).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        
        # 过去20天箱体指标
        df['box_high'] = df['最高'].rolling(20).shift(1).max()
        df['box_low'] = df['最低'].rolling(20).shift(1).min()
        df['box_amp'] = (df['box_high'] - df['box_low']) / df['box_low']
        df['avg_vol_20'] = df['成交量'].rolling(20).shift(1).mean()
        
        # 未来5天收益率（用于结算）
        df['future_return'] = (df['收盘'].shift(-5) - df['收盘']) / df['收盘'] * 100

        # --- 战法条件判定 (掩码向量) ---
        cond = (
            (df['收盘'] >= 5.0) & (df['收盘'] <= 20.0) &                # 价格
            (df['box_amp'] <= 0.12) &                                 # 振幅
            (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']) &    # 均线多头
            (df['涨跌幅'] >= 6.0) &                                    # 涨幅强度
            (df['收盘'] > df['box_high']) &                            # 突破箱体
            (df['成交量'] > df['avg_vol_20'] * 3.5) &                  # 量比
            (df['换手率'] >= 3.0) & (df['换手率'] <= 8.0) &            # 换手
            ((df['最高'] - df['收盘']) / df['收盘'] <= 0.02)           # 无长上影
        )
        
        # 提取符合条件的交易记录
        trades = df[cond][['日期', 'future_return']].dropna()
        if trades.empty: return None
        
        trades['year'] = trades['日期'].astype(str).str[:4]
        return trades[['year', 'future_return']]
    except:
        return None

def main():
    stock_data_path = './stock_data/*.csv'
    files = glob.glob(stock_data_path)
    print(f"🚀 启动并行回测引擎... 目标文件数: {len(files)}")

    # 使用所有可用 CPU 核心进行并行计算
    results = []
    with ProcessPoolExecutor() as executor:
        # map 保持顺序，利用多进程加速
        res_list = list(executor.map(process_single_file, files))
    
    # 清理并合并数据
    valid_dfs = [r for r in res_list if r is not None]
    if not valid_dfs:
        print("❌ 未发现任何符合战法信号的交易记录。")
        return

    all_trades = pd.concat(valid_dfs)
    
    # --- 统计报表 ---
    summary = all_trades.groupby('year')['future_return'].agg(
        交易次数='count',
        平均收益_pct='mean',
        胜率_pct=lambda x: (x > 0).sum() / len(x) * 100
    ).round(2)
    
    # 汇总
    total_row = pd.DataFrame({
        '交易次数': [len(all_trades)],
        '平均收益_pct': [all_trades['future_return'].mean()],
        '胜率_pct': [(all_trades['future_return'] > 0).sum() / len(all_trades) * 100]
    }, index=['所有年份汇总']).round(2)
    
    final_report = pd.concat([summary, total_row])
    
    # --- 保存结果 ---
    os.makedirs('backtest_results', exist_ok=True)
    report_path = f"backtest_results/summary_{datetime.now().strftime('%Y%m%d')}.csv"
    final_report.to_csv(report_path, encoding='utf-8-sig')
    
    print("\n" + "="*30)
    print(final_report)
    print("="*30)
    print(f"✅ 回测报告已保存至: {report_path}")

if __name__ == "__main__":
    main()
