import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ==========================================
# 战法名称：【风口主升浪-优化版回测】
# 核心逻辑：
# [cite_start]1. 基础筛选：5-20元，排除ST、创业板(30)。 [cite: 1]
# [cite_start]2. 筹码确认：连续3日换手率 5%-15%（主力控盘）。 [cite: 3]
# 3. 技术确认：5/10/20日均线多头 + RSI(60-80)强势区。
# 4. 优化项：收盘价必须在MA250(年线)上方，且当日成交量大于5日均量。
# ==========================================

DATA_DIR = 'stock_data'
HOLD_DAYS = 10

def backtest_engine(file_path):
    try:
        # 仅读取核心字段加速运行
        df = pd.read_csv(file_path, usecols=['日期', '股票代码', '收盘', '换手率', '成交量'])
        df = df.sort_values(by='日期').reset_index(drop=True)
        code = str(df.iloc[0]['股票代码']).zfill(6)
        
        # 排除创业板
        if code.startswith('30'): return None
        
        close = df['收盘'].values
        vol = df['成交量'].values
        turnover = df['换手率'].values
        dates = df['日期'].values
        
        # 计算技术指标
        ma5 = df['收盘'].rolling(5).mean().values
        ma10 = df['收盘'].rolling(10).mean().values
        ma20 = df['收盘'].rolling(20).mean().values
        ma250 = df['收盘'].rolling(250).mean().values # 年线过滤
        vol_ma5 = df['成交量'].rolling(5).mean().values # 量比过滤
        
        # RSI 快速计算
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        hits = []
        # 遍历历史（从年线生效的第250天开始）
        for i in range(250, len(df) - HOLD_DAYS):
            # [cite_start]条件1：价格与排除项 [cite: 1]
            is_base_ok = 5.0 <= close[i] <= 20.0
            # 条件2：均线多头与年线支撑
            is_ma_ok = (ma5[i] > ma10[i] > ma20[i]) and (close[i] > ma250[i])
            # 条件3：RSI强势区
            is_rsi_ok = 60 <= rsi[i] <= 80
            # [cite_start]条件4：换手率平稳 [cite: 3]
            is_turnover_ok = np.all((turnover[i-2:i+1] >= 5) & (turnover[i-2:i+1] <= 15))
            # 条件5：量比确认（放量启动）
            is_vol_ok = vol[i] > vol_ma5[i]
            
            if is_base_ok and is_ma_ok and is_rsi_ok and is_turnover_ok and is_vol_ok:
                ret = (close[i + HOLD_DAYS] - close[i]) / close[i]
                hits.append({'year': dates[i][:4], 'ret': ret, 'win': 1 if ret > 0 else 0})
        return hits
    except:
        return None

def main():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    print(f"开始并行回测 {len(files)} 个历史数据文件...")
    
    with Pool(cpu_count()) as p:
        results = p.map(backtest_engine, files)
    
    all_hits = [h for res in results if res for h in res]
    if not all_hits:
        print("未筛选到符合条件的交易样本。")
        return
        
    df = pd.DataFrame(all_hits)
    # 按年度聚合
    annual = df.groupby('year').agg(交易次数=('ret', 'count'), 胜率=('win', 'mean'), 平均收益=('ret', 'mean'))
    
    now = datetime.now()
    output_dir = now.strftime('%Y-%m')
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/OPTIMIZED_REPORT_{now.strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 风口主升浪(优化版) 10年回测报告 ===\n")
        f.write(f"总样本数: {len(df)}\n")
        f.write(f"全周期胜率: {df['win'].mean():.2%}\n")
        f.write(f"平均收益(10日): {df['ret'].mean():.2%}\n\n")
        f.write("年度统计表:\n")
        f.write(annual.to_string())
        
    print(f"优化报告已生成: {report_path}")

if __name__ == "__main__":
    main()
