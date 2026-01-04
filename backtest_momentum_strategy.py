import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==========================================
# 战法名称：【风口主升浪-10年历史回测版】
# 逻辑说明：筛选 10 年内符合“均线多头+RSI强势区+温和换手”的个股表现
# ==========================================

DATA_DIR = 'stock_data'
HOLD_DAYS = 10  # 统一以10日持有期为准进行统计

def calculate_rsi_fast(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / (down + 1e-9)
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval, downval = delta, 0.
        else:
            upval, downval = 0., -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / (down + 1e-9)
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def backtest_engine(file_path):
    try:
        # 仅读取必要列以节省内存
        df = pd.read_csv(file_path, usecols=['日期', '股票代码', '收盘', '换手率'])
        if len(df) < 60: return None
        
        df = df.sort_values(by='日期').reset_index(drop=True)
        code = str(df.iloc[0]['股票代码']).zfill(6)
        
        # 排除创业板 (30开头)
        if code.startswith('30'): return None

        close = df['收盘'].values
        turnover = df['换手率'].values
        dates = df['日期'].values
        
        # 计算技术指标
        ma5 = df['收盘'].rolling(5).mean().values
        ma10 = df['收盘'].rolling(10).mean().values
        ma20 = df['收盘'].rolling(20).mean().values
        rsi = calculate_rsi_fast(close)
        
        results = []
        # 核心扫描循环
        for i in range(20, len(df) - HOLD_DAYS):
            # 战法条件判断
            cond_price = 5.0 <= close[i] <= 20.0
            cond_ma = ma5[i] > ma10[i] > ma20[i]
            cond_rsi = 60 <= rsi[i] <= 80
            # 连续3日换手率在 5%-15%
            cond_turnover = np.all((turnover[i-2:i+1] >= 5) & (turnover[i-2:i+1] <= 15))
            
            if cond_price and cond_ma and cond_rsi and cond_turnover:
                ret = (close[i + HOLD_DAYS] - close[i]) / close[i]
                results.append({
                    'year': dates[i][:4],
                    'return': ret,
                    'is_win': 1 if ret > 0 else 0
                })
        return results
    except:
        return None

def main():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"检测到 {len(files)} 个数据文件，启动多核并行回测...")
    
    with Pool(cpu_count()) as p:
        raw_results = p.map(backtest_engine, files)
    
    # 平坦化处理结果
    all_trades = [t for res in raw_results if res for t in res]
    if not all_trades:
        print("未筛选到任何历史符合条件的交易。")
        return
        
    df_results = pd.DataFrame(all_trades)
    
    # 生成统计报告
    total_trades = len(df_results)
    win_rate = df_results['is_win'].mean()
    avg_ret = df_results['return'].mean()
    
    # 按年度统计
    annual_stats = df_results.groupby('year').agg(
        交易次数=('return', 'count'),
        胜率=('is_win', 'mean'),
        平均收益=('return', 'mean')
    )
    
    # 写入文件
    now = datetime.now()
    output_dir = now.strftime('%Y-%m')
    os.makedirs(output_dir, exist_ok=True)
    report_name = os.path.join(output_dir, f"TEN_YEAR_REPORT_{now.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_name, 'w', encoding='utf-8') as f:
        f.write(f"=== 风口主升浪战法：10年历史回测报告 ===\n")
        f.write(f"回测日期: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计触发交易次数: {total_trades}\n")
        f.write(f"全周期总胜率: {win_rate:.2%}\n")
        f.write(f"单笔平均收益 (10日持有): {avg_ret:.2%}\n")
        f.write("-" * 40 + "\n")
        f.write("年度分表统计:\n")
        f.write(annual_stats.to_string())
        f.write("\n" + "-" * 40 + "\n")
        f.write("备注：该回测仅包含深沪A股（非创业板），价格区间5-20元。\n")

    print(f"报告已生成完毕: {report_name}")

if __name__ == "__main__":
    main()
