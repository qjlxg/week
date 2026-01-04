import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 战法名称：【风口主升浪 - 趋势优化版】
# 核心指标：
# 1. 价格区间：5-20元；排除创业板(30开头)。
# 2. 筹码逻辑：连续3日换手率 5%-15%（主力锁仓）。
# 3. 技术形态：5/10/20日均线多头排列 + RSI(60-80)强势区。
# 4. 优化因子：股价高于MA250年线（大趋势向上）+ 买入当日放量（资金攻击）。
# ==============================================================================

DATA_DIR = 'stock_data'
HOLD_DAYS = 10  # 默认持有10个交易日进行收益统计

def strategy_engine(file_path):
    try:
        # 优化内存：仅读取必要字段
        df = pd.read_csv(file_path, usecols=['日期', '股票代码', '收盘', '换手率', '成交量'])
        df = df.sort_values(by='日期').reset_index(drop=True)
        
        code = str(df.iloc[0]['股票代码']).zfill(6)
        if code.startswith('30'): return None  # 排除创业板

        close = df['收盘'].values
        vol = df['成交量'].values
        turnover = df['换手率'].values
        dates = df['日期'].values
        
        # --- 计算技术指标 ---
        ma5 = df['收盘'].rolling(5).mean().values
        ma10 = df['收盘'].rolling(10).mean().values
        ma20 = df['收盘'].rolling(20).mean().values
        ma250 = df['收盘'].rolling(250).mean().values  # 优化项：趋势过滤
        vol_ma5 = df['成交量'].rolling(5).mean().values # 优化项：成交量过滤
        
        # 快速计算 RSI (14)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        results = []
        # 从指标生效期开始扫描
        for i in range(250, len(df) - HOLD_DAYS):
            # 条件1：价格与排除项
            cond_price = 5.0 <= close[i] <= 20.0
            # 条件2：均线多头排列 + 股价在年线之上
            cond_ma = (ma5[i] > ma10[i] > ma20[i]) and (close[i] > ma250[i])
            # 条件3：RSI处于强势区间
            cond_rsi = 60 <= rsi[i] <= 80
            # 条件4：主力锁仓（连续3日换手率 5%-15%）
            cond_turnover = np.all((turnover[i-2:i+1] >= 5) & (turnover[i-2:i+1] <= 15))
            # 条件5：量比确认（当日放量启动）
            cond_vol = vol[i] > vol_ma5[i]
            
            if cond_price and cond_ma and cond_rsi and cond_turnover and cond_vol:
                ret = (close[i + HOLD_DAYS] - close[i]) / close[i]
                results.append({
                    'year': dates[i][:4],
                    'ret': ret,
                    'win': 1 if ret > 0 else 0
                })
        return results
    except Exception:
        return None

def main():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"[{datetime.now()}] 启动并行扫描，样本库文件总数: {len(files)}")
    
    # 启动并行计算
    with Pool(cpu_count()) as p:
        raw_output = p.map(strategy_engine, files)
    
    # 平坦化处理结果
    all_trades = [t for res in raw_output if res for t in res]
    if not all_trades:
        print("未发现匹配“风口主升浪”战法的历史交易样本。")
        return
        
    df_results = pd.DataFrame(all_trades)
    
    # 生成统计报告
    annual_stats = df_results.groupby('year').agg(
        信号次数=('ret', 'count'),
        胜率=('win', 'mean'),
        平均收益率=('ret', 'mean')
    )
    
    # 输出与保存
    now = datetime.now()
    output_dir = now.strftime('%Y-%m')
    os.makedirs(output_dir, exist_ok=True)
    report_name = os.path.join(output_dir, f"STRATEGY_REPORT_{now.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_name, 'w', encoding='utf-8') as f:
        f.write(f"=== 风口主升浪战法：优化版回测汇总报告 ===\n")
        f.write(f"全周期胜率: {df_results['win'].mean():.2%}\n")
        f.write(f"总计触发样本: {len(df_results)}\n")
        f.write("-" * 45 + "\n年度明细：\n")
        f.write(annual_stats.to_string())
        
    print(annual_stats)
    print(f"\n报告已生成至: {report_name}")

if __name__ == "__main__":
    main()
