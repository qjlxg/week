import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import pytz
from itertools import product

# =====================================================================
#                       精细化参数寻优区间
# =====================================================================
PARAM_GRID = {
    'min_profit_potential': list(range(15, 41, 5)),      
    'rsi_max': list(range(15, 36, 2)),                   
    'hold_days': [5, 10, 15, 20, 25, 30],                 
    'stop_loss': [round(x, 3) for x in np.arange(-0.05, -0.21, -0.025)] 
}

DATA_DIR = "stock_data"
REPORT_DIR = "results"
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')

def fast_calculate_indicators(df):
    """仅计算基础数据，不带任何过滤参数"""
    if len(df) < 65: return None
    try:
        close = df['收盘'].values
        high = df['最高'].values
        low = df['最低'].values
        ma5 = pd.Series(close).rolling(5).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values
        potential = (ma60 - close) / np.where(close == 0, 1, close) * 100
        
        # RSI 快速计算
        delta = np.diff(close, prepend=close[0])
        up = pd.Series(np.where(delta > 0, delta, 0)).ewm(alpha=1/6, adjust=False).mean().values
        dn = pd.Series(np.where(delta < 0, -delta, 0)).ewm(alpha=1/6, adjust=False).mean().values
        rsi6 = 100 - (100 / (1 + (up / np.where(dn == 0, 1e-9, dn))))
        
        return {
            'close': close, 'low': low, 'high': high, 
            'rsi6': rsi6, 'potential': potential, 'ma5': ma5
        }
    except: return None

def main():
    start_time = datetime.now()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # 1. 预提取所有潜在的候选点 (只做最松的过滤)
    print(f"📦 正在扫描原始信号点...")
    raw_signals = []
    for f in files:
        ind = fast_calculate_indicators(pd.read_csv(f))
        if ind is None: continue
        
        # 找出所有 RSI < 40 且 空间 > 10% 的点作为原始池
        # 基础确认：收盘 > MA5 (确保不是在阴跌中)
        mask = (ind['rsi6'] < 40) & (ind['potential'] > 10) & (ind['close'] >= ind['ma5'])
        indices = np.where(mask)[0]
        
        for idx in indices:
            # 记录该点的特征，用于后续快速筛选
            # 记录未来 30 天的每日价格，方便计算不同 hold_days
            if idx + 31 >= len(ind['close']): continue
            
            raw_signals.append({
                'rsi': ind['rsi6'][idx],
                'pot': ind['potential'][idx],
                'price': ind['close'][idx],
                'future_lows': ind['low'][idx+1 : idx+31],
                'future_closes': ind['close'][idx+1 : idx+31]
            })

    if not raw_signals:
        print("❌ 未发现任何原始信号"); return

    # 转换为 NumPy 矩阵进行矢量化压榨
    print(f"⚡ 原始池构建完毕 ({len(raw_signals)}个点)，开始暴力寻优...")
    rsi_arr = np.array([s['rsi'] for s in raw_signals])
    pot_arr = np.array([s['pot'] for s in raw_signals])
    price_arr = np.array([s['price'] for s in raw_signals])
    f_lows = np.array([s['future_lows'] for s in raw_signals])   # Shape: (N, 30)
    f_closes = np.array([s['future_closes'] for s in raw_signals]) # Shape: (N, 30)

    results = []
    # 2. 嵌套循环寻优 (现在循环内部全是 NumPy 矩阵运算)
    keys = PARAM_GRID.keys()
    for p_pot, p_rsi, p_hold, p_stop in product(*PARAM_GRID.values()):
        # 一行代码筛选所有满足当前参数的点
        mask = (rsi_arr <= p_rsi) & (pot_arr >= p_pot)
        if not np.any(mask): continue
        
        # 提取选中点的未来表现
        sel_lows = f_lows[mask][:, :p_hold]
        sel_closes = f_closes[mask][:, p_hold-1]
        sel_prices = price_arr[mask]
        
        # 计算止损情况：每一行中是否有价格触及止损线
        low_returns = (sel_lows - sel_prices[:, None]) / sel_prices[:, None]
        is_stop = np.any(low_returns <= p_stop, axis=1)
        
        # 计算最终收益：止损的给 p_stop，没止损的给持有到期收益
        final_rets = np.where(is_stop, p_stop, (sel_closes - sel_prices) / sel_prices)
        
        results.append({
            'min_pot': p_pot, 'rsi_max': p_rsi, 'hold': p_hold, 'stop': p_stop,
            'count': len(final_rets),
            'win_rate': np.sum(final_rets > 0) / len(final_rets),
            'avg_ret': np.mean(final_rets)
        })

    # 3. 输出报表
    res_df = pd.DataFrame(results).sort_values('win_rate', ascending=False)
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"Fast_Opt_{datetime.now(SHANGHAI_TZ).strftime('%Y%m%d')}.md")
    res_df.head(50).to_markdown(report_path, index=False)
    
    print(f"✅ 耗时: {datetime.now() - start_time} | 最佳胜率: {res_df.iloc[0]['win_rate']:.2%}")

if __name__ == "__main__":
    main()
