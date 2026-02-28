import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count
from datetime import datetime
import pytz
from itertools import product

# =====================================================================
#                       参数寻优区间配置
# =====================================================================
PARAM_GRID = {
    'min_profit_potential': [20, 25, 30],      # 空间门槛微调
    'rsi_max': [20, 25, 30],                   # RSI超卖阈值
    'hold_days': [10, 20, 30],                 # 持仓天数微调
    'stop_loss': [-0.08, -0.10, -0.12]         # 止损线
}

DATA_DIR = "stock_data"
REPORT_DIR = "results"
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')

# =====================================================================
#                       核心计算引擎 (保持原逻辑不变)
# =====================================================================
def calculate_indicators(df):
    if df is None or len(df) < 65: return None
    # 按照您的CSV格式：日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
    try:
        close = df['收盘'].values
        high = df['最高'].values
        low = df['最低'].values
        vol = df['成交量'].values
        turnover = df['换手率'].values
    except: return None
    
    # 指标计算 (简化版，保留核心)
    ma5 = pd.Series(close).rolling(5).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values
    potential = (ma60 - close) / np.where(close == 0, 1, close) * 100
    
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    dn = np.where(delta < 0, -delta, 0)
    def rma(x, n): return pd.Series(x).ewm(alpha=1/n, adjust=False).mean().values
    rsi6 = 100 - (100 / (1 + (rma(up, 6) / np.where(rma(dn, 6) == 0, 1e-9, rma(dn, 6)))))
    
    ma5_change = (ma5 - np.roll(ma5, 1)) / np.where(ma5 == 0, 1, ma5)
    vol_ma5 = pd.Series(vol).shift(1).rolling(5).mean().values
    vol_ratio = vol / np.where(vol_ma5 == 0, 1e-9, vol_ma5)
    
    return {'close': close, 'low': low, 'rsi6': rsi6, 'potential': potential, 'vol_ratio': vol_ratio, 'ma5': ma5, 'ma5_change': ma5_change}

def run_strategy(ind, params):
    """带参数的信号判定"""
    close, low, rsi6, potential = ind['close'], ind['low'], ind['rsi6'], ind['potential']
    vol_ratio, ma5, ma5_c = ind['vol_ratio'], ind['ma5'], ind['ma5_change']
    
    # 基础过滤 (使用传入参数)
    mask = (potential >= params['min_profit_potential']) & (rsi6 <= params['rsi_max']) & (close >= ma5)
    return np.where(mask)[0]

# =====================================================================
#                       并行回测逻辑
# =====================================================================
def evaluate_combination(args):
    params, file_list = args
    all_rets = []
    
    for f in file_list:
        try:
            df = pd.read_csv(f)
            ind = calculate_indicators(df)
            if ind is None: continue
            
            sig_indices = run_strategy(ind, params)
            hold = params['hold_days']
            stop = params['stop_loss']
            
            for idx in sig_indices:
                if idx + hold >= len(ind['close']): continue
                entry_p = ind['close'][idx]
                p_low = ind['low'][idx+1 : idx+hold+1].min()
                
                if (p_low - entry_p) / entry_p <= stop:
                    all_rets.append(stop)
                else:
                    all_rets.append((ind['close'][idx+hold] - entry_p) / entry_p)
        except: continue
        
    if not all_rets: return None
    rets = np.array(all_rets)
    return {
        **params,
        'count': len(rets),
        'win_rate': np.sum(rets > 0) / len(rets),
        'avg_ret': np.mean(rets)
    }

def main():
    start_t = datetime.now()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # 生成参数组合
    keys = PARAM_GRID.keys()
    combinations = [dict(zip(keys, v)) for v in product(*PARAM_GRID.values())]
    print(f"🚀 开始寻优 | 组合数: {len(combinations)} | 样本数: {len(files)}")

    # 使用进程池加速
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(evaluate_combination, [(c, files) for c in combinations])
    
    results = [r for r in results if r]
    res_df = pd.DataFrame(results).sort_values('win_rate', ascending=False)
    
    # 保存结果
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_file = os.path.join(REPORT_DIR, f"Opt_Report_{datetime.now(SHANGHAI_TZ).strftime('%Y%m%d')}.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 📈 参数寻优结果 (上海时区)\n\n生成时间: {datetime.now(SHANGHAI_TZ)}\n\n")
        f.write(res_df.to_markdown(index=False))
        
    print(f"✅ 寻优完毕 | 最优胜率: {res_df.iloc[0]['win_rate']:.2%}")

if __name__ == "__main__":
    main()
