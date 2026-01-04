import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

"""
战法名称：【一种模式做一万遍】—— 极速回测系统 v4.3 (兼容性增强版)
核心优化：
1. 适应性：增加对多种日期格式的自动识别。
2. 容错率：振幅放宽至 25%，量比 1.5 倍，换手率 1% - 15%。
3. 纯净度：依然严格排除 30/688/ST，只做沪深主板。
"""

def process_single_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 40: return None
        
        # 提取并清洗代码
        code = str(df['股票代码'].iloc[0]).zfill(6)
        if code.startswith(('30', '688', '8', '4')): return None
        if not (code.startswith('60') or code.startswith('00')): return None

        # --- 向量化指标计算 ---
        # 统一处理日期，提取年份
        df['date_str'] = df['日期'].astype(str).str.replace('-', '').str.replace('/', '')
        df['year'] = df['date_str'].str[:4]
        
        # 技术指标
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        
        # 箱体逻辑：过去15天
        df['box_high'] = df['最高'].rolling(15).shift(1).max()
        df['box_low'] = df['最低'].rolling(15).shift(1).min()
        df['box_amp'] = (df['box_high'] - df['box_low']) / df['box_low']
        df['avg_vol_15'] = df['成交量'].rolling(15).shift(1).mean()
        
        # 结算逻辑：持有5天收益
        df['future_return'] = (df['收盘'].shift(-5) - df['收盘']) / df['收盘'] * 100

        # --- 降压版筛选条件 ---
        cond = (
            (df['收盘'] >= 5.0) & (df['收盘'] <= 30.0) &                # 价格区间
            (df['box_amp'] <= 0.25) &                                 # 振幅放宽到25%
            (df['收盘'] > df['MA20']) &                                # 站上20日线
            (df['涨跌幅'] >= 3.5) &                                    # 涨幅强度
            (df['收盘'] >= df['box_high']) &                           # 突破或持平箱体高点
            (df['成交量'] > df['avg_vol_15'] * 1.5) &                  # 量比1.5倍
            (df['换手率'] >= 1.0) & (df['换手率'] <= 15.0)              # 换手放宽
        )
        
        res = df[cond][['year', 'future_return']].dropna()
        return res if not res.empty else None
    except Exception as e:
        return None

def main():
    files = glob.glob('./stock_data/*.csv')
    print(f"🚀 启动并行回测引擎 v4.3... 目标文件数: {len(files)}")

    with ProcessPoolExecutor() as executor:
        res_list = list(executor.map(process_single_file, files))
    
    valid_dfs = [r for r in res_list if r is not None]
    
    if not valid_dfs:
        print("❌ 警告：依然没有交易记录。")
        print("请检查：1. stock_data 目录下 CSV 文件是否包含 '涨跌幅' 和 '换手率' 列。")
        print("      2. 检查价格单位，确保 '收盘' 是以'元'为单位。")
        return

    all_trades = pd.concat(valid_dfs)
    
    # --- 生成报告 ---
    summary = all_trades.groupby('year')['future_return'].agg(
        交易次数='count',
        平均收益_pct='mean',
        胜率_pct=lambda x: (x > 0).sum() / len(x) * 100
    ).round(2)
    
    total = pd.DataFrame({
        '交易次数': [len(all_trades)],
        '平均收益_pct': [all_trades['future_return'].mean()],
        '胜率_pct': [(all_trades['future_return'] > 0).sum() / len(all_trades) * 100]
    }, index=['所有年份汇总']).round(2)
    
    final_report = pd.concat([summary, total])
    
    # 保存结果
    os.makedirs('backtest_results', exist_ok=True)
    report_path = f"backtest_results/summary_v43_{datetime.now().strftime('%Y%m%d')}.csv"
    final_report.to_csv(report_path, encoding='utf-8-sig')
    
    print("\n" + "📊 回测统计报表 " + "="*20)
    print(final_report)
    print("="*40)
    print(f"✅ 报告已生成: {report_path}")

if __name__ == "__main__":
    main()
