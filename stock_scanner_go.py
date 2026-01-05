import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count
from datetime import datetime
import pytz

# =====================================================================
#                          核心参数区 (保持您的配置不变)
# =====================================================================
INDEX_CODE = 'sz000300'        
ENABLE_MARKET_FILTER = False   

MIN_PRICE = 5.0                
MAX_AVG_TURNOVER_30 = 2.5      
MIN_PROFIT_POTENTIAL = 25      # 空间门槛

RSI_MAX = 25                   
KDJ_K_MAX = 25                 
MAX_TODAY_CHANGE = 3.0         

SHRINK_VOL_MAX = 0.85          
ADD_POS_VOL_RATIO = 1.5        

STOP_LOSS_LIMIT = -0.10        # 初始硬止损
HOLD_DAYS = 20                 # 最大持仓天数

DATA_DIR = "stock_data"
REPORT_DIR = "results"
NAME_MAP_FILE = 'stock_names.csv' 
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')

# =====================================================================
#                          计算与判定逻辑 (指标计算部分不变)
# =====================================================================
def calculate_indicators(df):
    if df is None or len(df) < 65: return None
    d_col = next((c for c in ['日期', 'date', '时间'] if c in df.columns), None)
    if d_col: df = df.sort_values(d_col).reset_index(drop=True)
    try:
        close, high, low = df['收盘'].values, df['最高'].values, df['最低'].values
        vol, turnover = df['成交量'].values, df['换手率'].values
    except: return None
    
    delta = np.diff(close, prepend=close[0])
    up, dn = np.where(delta > 0, delta, 0), np.where(delta < 0, -delta, 0)
    def rma(x, n): return pd.Series(x).ewm(alpha=1/n, adjust=False).mean().values
    rsi6 = 100 - (100 / (1 + (rma(up, 6) / np.where(rma(dn, 6) == 0, 1e-9, rma(dn, 6)))))
    
    ma5 = pd.Series(close).rolling(5).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values 
    ma60 = pd.Series(close).rolling(60).mean().values 
    potential = (ma60 - close) / np.where(close == 0, 1, close) * 100
    
    ma5_change = (ma5 - np.roll(ma5, 1)) / np.where(ma5 == 0, 1, ma5)
    avg_amp_3 = pd.Series((high - low) / close).rolling(3).mean().values
    vol_ratio = vol / np.where(pd.Series(vol).shift(1).rolling(5).mean() == 0, 1e-9, pd.Series(vol).shift(1).rolling(5).mean())
    change = pd.Series(close).pct_change().values * 100
    avg_turnover_30 = pd.Series(turnover).rolling(30).mean().values

    return {
        'close': close, 'low': low, 'high': high, 'rsi6': rsi6, 'ma20': ma20, 'ma60': ma60, 
        'vol_ratio': vol_ratio, 'ma5_change': ma5_change, 'avg_amp_3': avg_amp_3, 
        'change': change, 'avg_turnover_30': avg_turnover_30, 'potential': potential, 'ma5': ma5
    }

def get_signals_fast(ind):
    # 此处逻辑保持与您之前完全一致，不做改动
    close, rsi6, ma5, vol_ratio, potential, change = ind['close'], ind['rsi6'], ind['ma5'], ind['vol_ratio'], ind['potential'], ind['change']
    basic_filter = (potential >= MIN_PROFIT_POTENTIAL) & (close >= MIN_PRICE) & (ind['avg_turnover_30'] <= MAX_AVG_TURNOVER_30) & (change <= MAX_TODAY_CHANGE)
    entry_confirm = (close >= ma5)
    sig_add = (np.roll(rsi6, 1) <= RSI_MAX) & (np.roll(vol_ratio, 1) <= SHRINK_VOL_MAX) & (vol_ratio >= ADD_POS_VOL_RATIO) & (change > 0) & entry_confirm & basic_filter
    return np.select([sig_add], ["🔥放量加仓"], default=None)

# =====================================================================
#                          改进版回测引擎 (解决盈亏比问题)
# =====================================================================
def backtest_task(file_path):
    try:
        df = pd.read_csv(file_path)
        ind = calculate_indicators(df)
        if ind is None: return None
        sigs = get_signals_fast(ind)
        indices = np.where(sigs != None)[0]
        
        trades = []
        for idx in indices:
            if idx + HOLD_DAYS >= len(ind['close']): continue
            
            entry_p = ind['close'][idx]
            current_sl = entry_p * (1 + STOP_LOSS_LIMIT) # 初始 -10% 止损
            is_closed = False
            
            for day in range(1, HOLD_DAYS + 1):
                curr_idx = idx + day
                h, l, m20, m60 = ind['high'][curr_idx], ind['low'][curr_idx], ind['ma20'][curr_idx], ind['ma60'][curr_idx]
                
                # 核心改进：移动止损逻辑
                # 如果最高价涨幅已经超过了 7%，将止损位提升到成本价 (保本位)
                # 这能有效防止“华立股份”这种票在获利后又跌回亏损
                if (h / entry_p - 1) >= 0.07:
                    current_sl = max(current_sl, entry_p)

                # 1. 监测止损 (可能是 -10%，也可能是保本位)
                if l <= current_sl:
                    trades.append((current_sl - entry_p) / entry_p)
                    is_closed = True; break
                
                # 2. 监测 MA60 终极止盈
                if h >= m60:
                    trades.append((m60 - entry_p) / entry_p)
                    is_closed = True; break
                
                # 3. 监测 MA20 增强止盈 (模仿实战：只有利润丰厚时才在 MA20 减仓)
                # 如果触碰 MA20 且利润已经超过 12%，执行止盈
                if h >= m20 and (h / entry_p - 1) >= 0.12:
                    trades.append((m20 - entry_p) / entry_p)
                    is_closed = True; break
            
            if not is_closed:
                trades.append((ind['close'][idx + HOLD_DAYS] - entry_p) / entry_p)
        return trades
    except: return None

# =====================================================================
#                          主程序
# =====================================================================
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"🧬 运行【移动止损+高门槛止盈版】| 样本: {len(files)}")
    
    with Pool(processes=cpu_count()) as pool:
        all_rets = [t for res in pool.map(backtest_task, files) if res for t in res]
    
    if all_rets:
        rets = np.array(all_rets)
        stats = f"总交易: {len(rets)} | 胜率: {np.sum(rets>0)/len(rets):.2%} | 平均收益: {np.mean(rets):.2%}"
        print(f"✅ 执行完毕 | {stats}")
    else:
        print("❌ 未发现交易记录")

if __name__ == "__main__":
    main()
