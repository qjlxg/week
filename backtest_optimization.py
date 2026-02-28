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
    'min_pot': [15, 20, 25, 30, 35, 40, 45],      
    'rsi_max': list(range(20, 41, 2)),                   
    'max_hold': [10, 15, 20, 25, 30, 35],                          
    'stop_loss': [round(x, 2) for x in np.arange(-0.07, -0.19, -0.01)],     
    'k_sell': [70, 75, 80, 85, 90]                 
}

DATA_DIR = "stock_data"
REPORT_DIR = "results"
MIN_TRADES = 500 
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')

def calculate_all_indicators(df):
    if len(df) < 65: return None
    try:
        # 针对您的CSV格式优化
        close = df['收盘'].values
        high = df['最高'].values
        low = df['最低'].values
        ma60 = pd.Series(close).rolling(60).mean().values
        pot = (ma60 - close) / np.where(close == 0, 1, close) * 100
        
        delta = np.diff(close, prepend=close[0])
        up = pd.Series(np.where(delta > 0, delta, 0)).ewm(alpha=1/6, adjust=False).mean().values
        dn = pd.Series(np.where(delta < 0, -delta, 0)).ewm(alpha=1/6, adjust=False).mean().values
        rsi6 = 100 - (100 / (1 + (up / np.where(dn == 0, 1e-9, dn))))
        
        l9, h9 = pd.Series(low).rolling(9).min(), pd.Series(high).rolling(9).max()
        rsv = (pd.Series(close) - l9) / (h9 - l9).replace(0, 1e-9) * 100
        k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
        d = pd.Series(k).ewm(com=2, adjust=False).mean().values
        
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd_h = ((ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()).values * 2

        return {'close':close, 'low':low, 'rsi6':rsi6, 'pot':pot, 'k':k, 'd':d, 'macd_h':macd_h}
    except: return None

def main():
    start_t = datetime.now()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files: return

    # 1. 预读取并提取所有基础信号
    raw_data = []
    print(f"📊 正在预载数据...")
    for f in files:
        ind = calculate_all_indicators(pd.read_csv(f))
        if ind is None: continue
        # 基础准入：RSI<45且空间>10且KDJ金叉
        mask = (ind['pot'] > 10) & (ind['rsi6'] < 45) & (ind['k'] > ind['d'])
        for idx in np.where(mask)[0]:
            if idx + 35 >= len(ind['close']): continue
            raw_data.append({
                'p': ind['close'][idx], 'pot': ind['pot'][idx], 'rsi': ind['rsi6'][idx],
                'fc': ind['close'][idx+1:idx+36], 'fl': ind['low'][idx+1:idx+36],
                'fk': ind['k'][idx+1:idx+36], 'fd': ind['d'][idx+1:idx+36], 'fm': ind['macd_h'][idx+1:idx+36]
            })

    if not raw_data: return
    
    # 转换为 NumPy 矩阵
    pot_v = np.array([x['pot'] for x in raw_data])
    rsi_v = np.array([x['rsi'] for x in raw_data])
    ep_v = np.array([x['p'] for x in raw_data])
    # 所有的未来走势矩阵化 (N, 35)
    fc_m = np.array([x['fc'] for x in raw_data])
    fl_m = np.array([x['fl'] for x in raw_data])
    fk_m = np.array([x['fk'] for x in raw_data])
    fd_m = np.array([x['fd'] for x in raw_data])
    fm_m = np.array([x['fm'] for x in raw_data])

    all_combos = list(product(*PARAM_GRID.values()))
    print(f"⚡ 开始寻优: {len(all_combos)} 组组合...")

    results = []
    for p_pot, p_rsi, p_hold, p_stop, p_k in all_combos:
        # 1. 快速筛选买入点
        m = (pot_v >= p_pot) & (rsi_v <= p_rsi)
        if np.sum(m) < MIN_TRADES: continue
        
        # 2. 矩阵逻辑运算：离场判定
        # 直接切片到当前 hold 周期
        s_fc, s_fl, s_fk, s_fd, s_fm = fc_m[m, :p_hold], fl_m[m, :p_hold], fk_m[m, :p_hold], fd_m[m, :p_hold], fm_m[m, :p_hold]
        s_ep = ep_v[m][:, None]
        
        # 离场标志位
        stop_trig = (s_fl - s_ep) / s_ep <= p_stop
        kdj_trig = (s_fk > p_k) & (s_fk < s_fd)
        macd_trig = (s_fm < 0)
        
        # 合并所有离场条件
        exit_mask = stop_trig | kdj_trig | macd_trig
        
        # --- 核心黑科技：使用 np.argmax 找到第一个 True 的位置 ---
        # 如果没有一个 True，argmax 会返回 0，所以需要配合 np.any
        has_exit = np.any(exit_mask, axis=1)
        exit_idx = np.argmax(exit_mask, axis=1)
        
        # 3. 计算收益率
        # 默认持有到期末
        rets = (s_fc[:, -1] - s_ep[:, 0]) / s_ep[:, 0]
        
        # 针对提前离场的点进行修正
        if np.any(has_exit):
            rows = np.where(has_exit)[0]
            cols = exit_idx[has_exit]
            # 止损触发的收益设为硬止损值，其他的设为离场当天收盘价收益
            is_stop = stop_trig[rows, cols]
            exit_rets = (s_fc[rows, cols] - s_ep[rows, 0]) / s_ep[rows, 0]
            rets[has_exit] = np.where(is_stop, p_stop, exit_rets)

        results.append([p_pot, p_rsi, p_hold, p_stop, p_k, len(rets), np.sum(rets>0)/len(rets), np.mean(rets)])

    # 4. 报表输出
    df = pd.DataFrame(results, columns=['空间','RSI','持仓','止损','KDJ顶','次数','胜率','均益']).sort_values('胜率', ascending=False)
    os.makedirs(REPORT_DIR, exist_ok=True)
    df.head(100).to_markdown(os.path.join(REPORT_DIR, f"Final_Opt_{datetime.now(SHANGHAI_TZ).strftime('%Y%m%d')}.md"), index=False)
    print(f"✅ 完成! 耗时: {datetime.now()-start_t} | 最佳胜率: {df.iloc[0]['胜率']:.2%}")

if __name__ == "__main__": main()
