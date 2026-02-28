import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import pytz
from itertools import product

# =====================================================================
#                       精细化参数寻优区间 (V2.3 极速分批版)
# =====================================================================
PARAM_GRID = {
    'min_pot': [15, 20, 25, 30, 35, 40, 45],      
    'rsi_max': list(range(20, 41, 2)),                   
    'max_hold': [10, 15, 20, 25, 30, 35],                          
    'stop_loss': [round(x, 2) for x in np.arange(-0.07, -0.19, -0.01)],     
    'k_sell': [70, 75, 80, 85, 90]                 
}

DATA_DIR = "stock_data"
REPORT_DIR = "results"
MIN_TRADES_FILTER = 500 
BATCH_SIZE = 500  # 每 500 组组合清理一次内存，防止卡死
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')

# =====================================================================
#                       全指标计算引擎
# =====================================================================
def calculate_all_indicators(df):
    if len(df) < 65: return None
    try:
        close = df['收盘'].values
        high = df['最高'].values
        low = df['最低'].values
        ma60 = pd.Series(close).rolling(60).mean().values
        potential = (ma60 - close) / np.where(close == 0, 1, close) * 100
        
        # RSI 快速算
        delta = np.diff(close, prepend=close[0])
        up = pd.Series(np.where(delta > 0, delta, 0)).ewm(alpha=1/6, adjust=False).mean().values
        dn = pd.Series(np.where(delta < 0, -delta, 0)).ewm(alpha=1/6, adjust=False).mean().values
        rsi6 = 100 - (100 / (1 + (up / np.where(dn == 0, 1e-9, dn))))
        
        # KDJ (9,3,3)
        l9, h9 = pd.Series(low).rolling(9).min(), pd.Series(high).rolling(9).max()
        rsv = (pd.Series(close) - l9) / (h9 - l9).replace(0, 1e-9) * 100
        k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
        d = pd.Series(k).ewm(com=2, adjust=False).mean().values
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        macd_h = (dif - dif.ewm(span=9, adjust=False).mean()).values * 2

        return {'close':close, 'low':low, 'rsi6':rsi6, 'potential':potential, 'k':k, 'd':d, 'macd_h':macd_h}
    except: return None

def main():
    start_time = datetime.now()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"📦 正在提取原始信号池...")
    raw_list = []
    for f in files:
        ind = calculate_all_indicators(pd.read_csv(f))
        if ind is None: continue
        mask = (ind['potential'] > 12) & (ind['rsi6'] < 42) & (ind['k'] > ind['d'])
        for idx in np.where(mask)[0]:
            if idx + 36 >= len(ind['close']): continue
            raw_list.append({
                'e_p': ind['close'][idx], 'pot': ind['potential'][idx], 'rsi': ind['rsi6'][idx],
                'fc': ind['close'][idx+1:idx+36], 'fl': ind['low'][idx+1:idx+36],
                'fk': ind['k'][idx+1:idx+36], 'fd': ind['d'][idx+1:idx+36], 'fm': ind['macd_h'][idx+1:idx+36]
            })

    if not raw_list: return
    
    # 转换为 NumPy 缓存
    pot = np.array([x['pot'] for x in raw_list])
    rsi = np.array([x['rsi'] for x in raw_list])
    ep = np.array([x['e_p'] for x in raw_list])
    fc, fl, fk, fd, fm = [np.array([x[k] for x in raw_list]) for k in ['fc','fl','fk','fd','fm']]
    
    all_combos = list(product(*PARAM_GRID.values()))
    total = len(all_combos)
    print(f"⚡ 信号池: {len(raw_list)}点 | 开启 {total} 组寻优...")

    final_res = []
    # --- 分批处理循环 ---
    for i in range(0, total, BATCH_SIZE):
        batch = all_combos[i : i + BATCH_SIZE]
        for p_pot, p_rsi, p_hold, p_stop, p_k_lvl in batch:
            idx_mask = (pot >= p_pot) & (rsi <= p_rsi)
            if np.sum(idx_mask) < MIN_TRADES_FILTER: continue
            
            # 提取子集
            sub_ep = ep[idx_mask][:, None]
            sub_fl, sub_fc, sub_fk, sub_fd, sub_fm = fl[idx_mask][:,:p_hold], fc[idx_mask][:,:p_hold], fk[idx_mask][:,:p_hold], fd[idx_mask][:,:p_hold], fm[idx_mask][:,:p_hold]
            
            # 逻辑判定
            stop_m = (sub_fl - sub_ep) / sub_ep <= p_stop
            kdj_m = (sub_fk > p_k_lvl) & (sub_fk < sub_fd)
            macd_m = (sub_fm < 0)
            
            exit_m = stop_m | kdj_m | macd_m
            has_ex = np.any(exit_m, axis=1)
            first_ex = np.argmax(exit_m, axis=1)
            
            # 计算收益
            rets = (sub_fc[:, p_hold-1] - ep[idx_mask]) / ep[idx_mask]
            if np.any(has_ex):
                row_idx = np.where(has_ex)[0]
                col_idx = first_ex[has_ex]
                # 止损位覆盖
                is_stop = stop_m[row_idx, col_idx]
                actual = (sub_fc[row_idx, col_idx] - ep[idx_mask][row_idx]) / ep[idx_mask][row_idx]
                rets[has_ex] = np.where(is_stop, p_stop, actual)

            final_res.append([p_pot, p_rsi, p_hold, p_stop, p_k_lvl, len(rets), np.sum(rets>0)/len(rets), np.mean(rets)])
        
        if i % 5000 == 0: print(f"⏳ 已完成 {i}/{total}...")

    # 输出结果
    df_res = pd.DataFrame(final_res, columns=['空间','RSI','持仓','止损','KDJ顶','次数','胜率','均益']).sort_values('胜率', ascending=False)
    df_res.head(100).to_markdown(os.path.join(REPORT_DIR, f"Fast_V23_{datetime.now(SHANGHAI_TZ).strftime('%Y%m%d')}.md"), index=False)
    print(f"✅ 耗时: {datetime.now()-start_time} | 最佳: {df_res.iloc[0]['胜率']:.2%}")

if __name__ == "__main__": main()
