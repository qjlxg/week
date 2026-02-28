import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import pytz
from itertools import product

# =====================================================================
#                       精细化参数寻优区间 (V2.2 深度细分)
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
MIN_TRADES_FILTER = 500  # 过滤掉交易次数太少的偶然性组合
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')

# =====================================================================
#                       全指标计算引擎
# =====================================================================
def calculate_all_indicators(df):
    if len(df) < 65: return None
    try:
        # 兼容你的CSV格式
        close = df['收盘'].values
        high = df['最高'].values
        low = df['最低'].values
        
        # 1. 空间与 RSI6
        ma60 = pd.Series(close).rolling(60).mean().values
        potential = (ma60 - close) / np.where(close == 0, 1, close) * 100
        delta = np.diff(close, prepend=close[0])
        up = pd.Series(np.where(delta > 0, delta, 0)).ewm(alpha=1/6, adjust=False).mean().values
        dn = pd.Series(np.where(delta < 0, -delta, 0)).ewm(alpha=1/6, adjust=False).mean().values
        rsi6 = 100 - (100 / (1 + (up / np.where(dn == 0, 1e-9, dn))))
        
        # 2. KDJ (9, 3, 3)
        l9 = pd.Series(low).rolling(9).min()
        h9 = pd.Series(high).rolling(9).max()
        rsv = (pd.Series(close) - l9) / (h9 - l9).replace(0, 1e-9) * 100
        k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
        d = pd.Series(k).ewm(com=2, adjust=False).mean().values
        
        # 3. MACD (12, 26, 9)
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd_h = ((ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()) * 2

        return {
            'close': close, 'low': low, 'rsi6': rsi6, 'potential': potential,
            'k': k, 'd': d, 'macd_h': macd_h.values
        }
    except: return None

# =====================================================================
#                           主逻辑
# =====================================================================
def main():
    start_time = datetime.now()
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    print(f"📦 正在扫描原始信号点 (包含 KDJ/MACD)...")
    raw_signals = []
    for f in files:
        ind = calculate_all_indicators(pd.read_csv(f))
        if ind is None: continue
        
        # 买入基础过滤：空间>10 & RSI<45 & KDJ金叉(K>D)
        # 预筛选稍微放宽，交给寻优器去收紧
        mask = (ind['potential'] > 10) & (ind['rsi6'] < 45) & (ind['k'] > ind['d'])
        indices = np.where(mask)[0]
        
        for idx in indices:
            if idx + 36 >= len(ind['close']): continue
            raw_signals.append({
                'entry_p': ind['close'][idx],
                'pot': ind['potential'][idx],
                'rsi': ind['rsi6'][idx],
                'f_close': ind['close'][idx+1 : idx+36],
                'f_low': ind['low'][idx+1 : idx+36],
                'f_k': ind['k'][idx+1 : idx+36],
                'f_d': ind['d'][idx+1 : idx+36],
                'f_macd_h': ind['macd_h'][idx+1 : idx+36]
            })

    if not raw_signals:
        print("❌ 未发现任何可用信号"); return

    # 转换为 NumPy 矩阵（寻优心脏）
    pot_arr = np.array([s['pot'] for s in raw_signals])
    rsi_arr = np.array([s['rsi'] for s in raw_signals])
    entry_arr = np.array([s['entry_p'] for s in raw_signals])
    f_close = np.array([s['f_close'] for s in raw_signals])
    f_low = np.array([s['f_low'] for s in raw_signals])
    f_k = np.array([s['f_k'] for s in raw_signals])
    f_d = np.array([s['f_d'] for s in raw_signals])
    f_macd_h = np.array([s['f_macd_h'] for s in raw_signals])

    print(f"⚡ 信号池: {len(raw_signals)}个点 | 开启 27,720 组参数暴力寻优...")
    
    results = []
    keys = PARAM_GRID.keys()
    # 使用笛卡尔积遍历所有组合
    for p_pot, p_rsi, p_hold, p_stop, p_k_lvl in product(*PARAM_GRID.values()):
        # 筛选满足当前买入条件的索引
        mask = (pot_arr >= p_pot) & (rsi_arr <= p_rsi)
        if np.sum(mask) < MIN_TRADES_FILTER: continue
        
        # 提取选中样本的未来走势矩阵
        m_entry = entry_arr[mask][:, None]
        m_low = f_low[mask][:, :p_hold]
        m_close = f_close[mask][:, :p_hold]
        m_k = f_k[mask][:, :p_hold]
        m_d = f_d[mask][:, :p_hold]
        m_macd = f_macd_h[mask][:, :p_hold]
        
        # 计算离场矩阵 (任一条件满足即离场)
        # 1. 硬止损线
        c1 = (m_low - m_entry) / m_entry <= p_stop
        # 2. KDJ 逃顶 (K值超过阈值且死叉)
        c2 = (m_k > p_k_lvl) & (m_k < m_d)
        # 3. MACD 转弱 (红柱转绿)
        c3 = (m_macd < 0)
        
        exit_matrix = c1 | c2 | c3
        
        # 矢量化提取每笔交易的最终收益
        # 找到每行第一个 True 的位置
        has_exit = np.any(exit_matrix, axis=1)
        first_exit = np.argmax(exit_matrix, axis=1)
        
        # 默认收益：如果没触发离场，按 max_hold 天收盘价计
        final_rets = (m_close[:, p_hold-1] - entry_arr[mask]) / entry_arr[mask]
        
        # 覆盖触发离场的情况
        if np.any(has_exit):
            # 获取触发离场当天的价格 (简便起见，止盈按收盘价，止损按 stop 价)
            exit_prices = m_close[np.arange(len(first_exit))[has_exit], first_exit[has_exit]]
            # 如果是止损触发的，收益强制设为 p_stop
            # 这里做一个近似处理：取 exit_idx 那天的 stop 判定
            is_stop_trigger = c1[np.arange(len(first_exit))[has_exit], first_exit[has_exit]]
            
            actual_rets = (exit_prices - entry_arr[mask][has_exit]) / entry_arr[mask][has_exit]
            # 修正止损回报
            actual_rets = np.where(is_stop_trigger, p_stop, actual_rets)
            final_rets[has_exit] = actual_rets

        results.append({
            '空间': p_pot, 'RSI': p_rsi, '持仓': p_hold, '止损': p_stop, 'KDJ顶': p_k_lvl,
            '次数': len(final_rets),
            '胜率': round(np.sum(final_rets > 0) / len(final_rets), 4),
            '均益': round(np.mean(final_rets), 4)
        })

    # 结果排序与输出
    res_df = pd.DataFrame(results).sort_values('胜率', ascending=False)
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_file = os.path.join(REPORT_DIR, f"Final_Opt_{datetime.now(SHANGHAI_TZ).strftime('%Y%m%d')}.md")
    res_df.head(100).to_markdown(out_file, index=False)
    
    print(f"✅ 任务完成！耗时: {datetime.now() - start_time}")
    print(f"📊 报告已生成: {out_file}")
    print(f"🏆 最佳组合胜率: {res_df.iloc[0]['胜率']:.2%}")

if __name__ == "__main__":
    main()
