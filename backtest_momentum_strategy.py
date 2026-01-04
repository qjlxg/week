import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 战法名称：【风口主升浪筛选法 - 10年回测优化版】
# 操作要领：
# 1. 选对风口：卡位国家政策热点（新质生产力、AI等）。
# 2. 盯准资金：换手率连续3天在5%-15%之间，主力资金净流入。
# 3. 把准技术：5/10/20日均线多头排列，RSI处于60-80强势区。
# 4. 优化条件：收盘价站在MA250年线上方（趋势过滤），当日放量（资金确认）。
# 5. 买卖逻辑：突破进场，跌破10日均线坚决离场。
# ==============================================================================

DATA_DIR = 'stock_data'
HOLD_DAYS = 10  # 统计买入后10个交易日的表现

def backtest_engine(file_path):
    """
    单只股票回测核心引擎
    """
    try:
        # 优化内存：仅读取回测核心字段
        df = pd.read_csv(file_path, usecols=['日期', '股票代码', '收盘', '换手率', '成交量'])
        df = df.sort_values(by='日期').reset_index(drop=True)
        
        # 排除创业板 (30开头)
        code = str(df.iloc[0]['股票代码']).zfill(6)
        if code.startswith('30'):
            return None
        
        close = df['收盘'].values
        vol = df['成交量'].values
        turnover = df['换手率'].values
        dates = df['日期'].values
        
        # --- 计算技术指标 ---
        ma5 = df['收盘'].rolling(5).mean().values
        ma10 = df['收盘'].rolling(10).mean().values
        ma20 = df['收盘'].rolling(20).mean().values
        ma250 = df['收盘'].rolling(250).mean().values  # 优化项：年线
        vol_ma5 = df['成交量'].rolling(5).mean().values # 优化项：5日均量
        
        # 快速计算 RSI (14)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        trade_logs = []
        # 从指标生效期（250日）开始扫描买点
        for i in range(250, len(df) - HOLD_DAYS):
            # 1. 价格过滤：5.0 - 20.0 元
            cond_price = 5.0 <= close[i] <= 20.0
            
            # 2. 均线多头排列 + 年线趋势过滤
            cond_ma = (ma5[i] > ma10[i] > ma20[i]) and (close[i] > ma250[i])
            
            # 3. RSI 强势区 (60-80)
            cond_rsi = 60 <= rsi[i] <= 80
            
            # 4. 换手率过滤：连续3日 5%-15%
            cond_turnover = np.all((turnover[i-2:i+1] >= 5) & (turnover[i-2:i+1] <= 15))
            
            # 5. 资金强度：买入当日放量 (成交量 > 5日均量)
            cond_vol = vol[i] > vol_ma5[i]
            
            if cond_price and cond_ma and cond_rsi and cond_turnover and cond_vol:
                # 模拟买入：计算10个交易日后的收益率
                ret = (close[i + HOLD_DAYS] - close[i]) / close[i]
                trade_logs.append({
                    'year': dates[i][:4],
                    'ret': ret,
                    'win': 1 if ret > 0 else 0
                })
        return trade_logs
    except Exception:
        return None

def main():
    # 1. 获取所有CSV文件
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"[{datetime.now()}] 启动并行回测，扫描文件数: {len(files)}")
    
    # 2. 多核并行运行加速
    with Pool(cpu_count()) as p:
        results = p.map(backtest_engine, files)
    
    # 3. 结果聚合
    all_trades = [t for res in results if res for t in res]
    if not all_trades:
        print("未发现匹配战法的历史交易。")
        return
        
    df_results = pd.DataFrame(all_trades)
    
    # 4. 生成统计分析报告
    annual_summary = df_results.groupby('year').agg(
        信号次数=('ret', 'count'),
        胜率=('win', 'mean'),
        平均收益率=('ret', 'mean')
    )
    
    # 5. 保存结果到年月目录
    now = datetime.now()
    dir_name = now.strftime('%Y-%m')
    os.makedirs(dir_name, exist_ok=True)
    report_file = os.path.join(dir_name, f"OPTIMIZED_BACKTEST_{now.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"=== 风口主升浪战法：10年优化版回测汇总 ===\n")
        f.write(f"触发时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总样本数: {len(df_results)}\n")
        f.write(f"全周期胜率: {df_results['win'].mean():.2%}\n")
        f.write(f"单笔平均收益 (10日持有): {df_results['ret'].mean():.2%}\n")
        f.write("-" * 45 + "\n")
        f.write("年度分表统计:\n")
        f.write(annual_summary.to_string())
        f.write("\n" + "-" * 45 + "\n")
        f.write("复盘逻辑：\n")
        f.write("1. 重点观察牛市年份（如2007/2015）与震荡年（2024/2025）的胜率差异 。\n")
        f.write("2. 优化后过滤了年线下方及缩量个股，理论胜率应高于基准版 。\n")

    print(f"回测报告已生成: {report_file}")

if __name__ == "__main__":
    main()
