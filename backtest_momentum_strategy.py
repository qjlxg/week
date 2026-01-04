import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 战法名称：【风口主升浪 - 安全参数回测版】
# 逻辑优化点：
# 1. RSI区间锁定：60 - 70（规避75+超买区，寻找主升浪初期标的）。
# 2. 乖离率控制：股价距离5日均线不超过 5%（防止追高）。
# 3. 量比控制：1.2 - 2.5 倍（温和放量，排除巨量见顶风险）。
# 4. 趋势基石：必须在 MA250 年线上方，且 5/10/20 均线多头。
# ==============================================================================

DATA_DIR = 'stock_data'
HOLD_DAYS = 10 

def backtest_engine(file_path):
    try:
        df = pd.read_csv(file_path, usecols=['日期', '股票代码', '收盘', '换手率', '成交量'])
        df = df.sort_values(by='日期').reset_index(drop=True)
        
        code = str(df.iloc[0]['股票代码']).zfill(6)
        if code.startswith('30'): return None # 依然排除创业板

        close = df['收盘'].values
        vol = df['成交量'].values
        turnover = df['换手率'].values
        dates = df['日期'].values
        
        # 指标计算
        ma5 = df['收盘'].rolling(5).mean().values
        ma10 = df['收盘'].rolling(10).mean().values
        ma20 = df['收盘'].rolling(20).mean().values
        ma250 = df['收盘'].rolling(250).mean().values
        vol_ma5 = df['成交量'].rolling(5).mean().values
        
        # RSI 计算
        delta = np.diff(close, prepend=close[0])
        avg_gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(14).mean().values
        avg_loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(14).mean().values
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        results = []
        for i in range(250, len(df) - HOLD_DAYS):
            # 1. 价格与均线多头 (含年线)
            cond_ma = (5.0 <= close[i] <= 20.0) and (ma5[i] > ma10[i] > ma20[i]) and (close[i] > ma250[i])
            
            # 2. RSI 安全强度区间 (60-70)
            cond_rsi = 60 <= rsi[i] <= 70
            
            # 3. 换手率锁仓 (连续3日 5%-15%)
            cond_turnover = np.all((turnover[i-2:i+1] >= 5) & (turnover[i-2:i+1] <= 15))
            
            # 4. 乖离率控制 (距5日线不远)
            cond_bias = close[i] <= ma5[i] * 1.05
            
            # 5. 量比温和放大 (1.2 - 2.5倍)
            vol_ratio = vol[i] / vol_ma5[i]
            cond_vol = 1.2 <= vol_ratio <= 2.5
            
            if cond_ma and cond_rsi and cond_turnover and cond_bias and cond_vol:
                ret = (close[i + HOLD_DAYS] - close[i]) / close[i]
                results.append({
                    'year': dates[i][:4],
                    'ret': ret,
                    'win': 1 if ret > 0 else 0
                })
        return results
    except:
        return None

def main():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"[{datetime.now()}] 开始安全参数版回测 (RSI 60-70)...")
    
    with Pool(cpu_count()) as p:
        raw_output = p.map(backtest_engine, files)
    
    all_trades = [t for res in raw_output if res for t in res]
    if not all_trades:
        print("未发现匹配信号。")
        return
        
    df_results = pd.DataFrame(all_trades)
    annual_stats = df_results.groupby('year').agg(
        信号次数=('ret', 'count'),
        胜率=('win', 'mean'),
        平均收益率=('ret', 'mean')
    )
    
    # 自动保存报告
    now = datetime.now()
    output_dir = now.strftime('%Y-%m')
    os.makedirs(output_dir, exist_ok=True)
    report_name = f"{output_dir}/SAFE_BACKTEST_{now.strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_name, 'w', encoding='utf-8') as f:
        f.write(f"=== 风口主升浪：安全参数版回测汇总 ===\n")
        f.write(f"逻辑：RSI(60-70) + 5日线乖离率<5% + 温和放量\n")
        f.write(f"全周期胜率: {df_results['win'].mean():.2%}\n")
        f.write(f"总样本数: {len(df_results)}\n")
        f.write("-" * 45 + "\n")
        f.write(annual_stats.to_string())
        
    print(annual_stats)
    print(f"\n报告已生成: {report_name}")

if __name__ == "__main__":
    main()
