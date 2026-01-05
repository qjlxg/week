import pandas as pd
import numpy as np
import os
import glob
import akshare as ak
from multiprocessing import Pool, cpu_count
from datetime import datetime
import pytz

# =====================================================================
#                          核心参数区 (盈利模型参数)
# =====================================================================
INDEX_CODE = 'sz000300'        
ENABLE_MARKET_FILTER = False   # 超跌策略在弱市中往往能捕捉到极致乖离机会

# --- 选股门槛 ---
MIN_PRICE = 5.0                
MAX_AVG_TURNOVER_30 = 2.5      
# 空间门槛：要求现价距离MA60至少有25%的距离，确保足够的“获利跑道”
MIN_PROFIT_POTENTIAL = 25      

# --- 技术指标阈值 ---
RSI_MAX = 25                   # 强力超跌区边界
KDJ_K_MAX = 25                 
MAX_TODAY_CHANGE = 3.0         # 限制涨幅，防止买在第一根阳线的末端

# --- 量能确认 ---
SHRINK_VOL_MAX = 0.85          # 缩量：代表抛压枯竭
ADD_POS_VOL_RATIO = 1.5        # 放量：代表主力回补确认

# --- 交易执行 (核心获利配置) ---
STOP_LOSS_LIMIT = -0.10        # 硬性止损：适配超跌股的大幅波动
HOLD_DAYS = 20                 # 持仓天数：给筑底和反弹留足20个交易日的时间

DATA_DIR = "stock_data"
REPORT_DIR = "results"
NAME_MAP_FILE = 'stock_names.csv' 
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')

# =====================================================================
#                          指标计算引擎
# =====================================================================

def calculate_indicators(df):
    """
    指标说明：
    1. RSI6 & KDJ_K: 判断超卖程度。
    2. MA5_Change: 判断5日均线是否止跌走平。
    3. Avg_Amp_3: 计算近3日平均振幅，判断K线是否收敛变小。
    4. Potential: 核心空间指标，计算当前价格回归60日线的理论涨幅。
    """
    if df is None or len(df) < 65: return None
    
    # 统一日期升序排列
    d_col = next((c for c in ['日期', 'date', '时间'] if c in df.columns), None)
    if d_col: df = df.sort_values(d_col).reset_index(drop=True)
    
    try:
        close = df['收盘'].values if '收盘' in df.columns else df['close'].values
        high = df['最高'].values if '最高' in df.columns else df['high'].values
        low = df['最低'].values if '最低' in df.columns else df['low'].values
        vol = df['成交量'].values if '成交量' in df.columns else df['volume'].values
        turnover = df['换手率'].values if '换手率' in df.columns else np.zeros(len(df))
    except: return None
    
    # RSI6 矢量化计算
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    dn = np.where(delta < 0, -delta, 0)
    def rma(x, n): return pd.Series(x).ewm(alpha=1/n, adjust=False).mean().values
    rsi6 = 100 - (100 / (1 + (rma(up, 6) / np.where(rma(dn, 6) == 0, 1e-9, rma(dn, 6)))))
    
    # KDJ (9,3,3)
    low_9 = pd.Series(low).rolling(9).min().values
    high_9 = pd.Series(high).rolling(9).max().values
    rsv = (close - low_9) / np.where(high_9 - low_9 == 0, 1e-9, high_9 - low_9) * 100
    kdj_k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
    
    # 均线系统与空间计算
    ma5 = pd.Series(close).rolling(5).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values
    potential = (ma60 - close) / np.where(close == 0, 1, close) * 100
    
    # 筑底识别：5日线变动率 + 近3日平均振幅
    ma5_change = (ma5 - np.roll(ma5, 1)) / np.where(ma5 == 0, 1, ma5)
    amplitude = (high - low) / np.where(close == 0, 1, close)
    avg_amp_3 = pd.Series(amplitude).rolling(3).mean().values
    
    # 量能分析
    vol_ma5 = pd.Series(vol).shift(1).rolling(5).mean().values
    vol_ratio = vol / np.where(vol_ma5 == 0, 1e-9, vol_ma5)
    
    # 辅助过滤
    min_3d_low = pd.Series(low).shift(1).rolling(3).min().values
    change = pd.Series(close).pct_change().values * 100
    avg_turnover_30 = pd.Series(turnover).rolling(30).mean().values

    return {
        'close': close, 'low': low, 'high': high, 'rsi6': rsi6, 'kdj_k': kdj_k,
        'ma5': ma5, 'ma60': ma60, 'vol_ratio': vol_ratio, 'ma5_change': ma5_change,
        'avg_amp_3': avg_amp_3, 'min_3d_low': min_3d_low, 'change': change, 
        'avg_turnover_30': avg_turnover_30, 'potential': potential
    }

# =====================================================================
#                          信号判定核心 (空间约束增强)
# =====================================================================

def get_signals_fast(ind):
    """
    逻辑定义：
    1. 💎地量筑底：强调K线变小(amp3)和均线止跌(ma5_c)，捕捉走平后的右侧转折。
    2. 🚀底部放量：强调极致超跌后的第一根温和放量阳线。
    3. 🔥放量加仓：强调在昨日超跌缩量的基础上，今日强力突破。
    """
    close, low, rsi6, kdj_k = ind['close'], ind['low'], ind['rsi6'], ind['kdj_k']
    ma5, vol_ratio, potential = ind['ma5'], ind['vol_ratio'], ind['potential']
    ma5_c, amp3 = ind['ma5_change'], ind['avg_amp_3']
    min_3d, change, turn30 = ind['min_3d_low'], ind['change'], ind['avg_turnover_30']
    
    # 统一基础过滤器：确保所有信号都必须具备 25% 以上的反弹空间
    basic_filter = (potential >= MIN_PROFIT_POTENTIAL) & (close >= MIN_PRICE) & \
                   (turn30 <= MAX_AVG_TURNOVER_30) & (change <= MAX_TODAY_CHANGE)

    # 共通入场确认：收盘不破5日线且不破前低
    entry_confirm = (close >= ma5) & (low >= min_3d)

    # --- A：💎地量筑底 (您的发现：均线走平，K线变小) ---
    sig_base_build = (rsi6 <= 30) & (ma5_c >= -0.005) & (amp3 <= 0.025) & \
                     (vol_ratio <= 1.0) & entry_confirm & basic_filter

    # --- B：🚀底部放量 (经典超跌反弹) ---
    sig_break = (rsi6 <= RSI_MAX) & (kdj_k <= KDJ_K_MAX) & (vol_ratio > SHRINK_VOL_MAX) & \
                (vol_ratio <= 2.0) & (change > 0) & entry_confirm & basic_filter
                
    # --- C：🔥放量加仓 (已修正：加入空间限制防止追高) ---
    # 逻辑：必须在满足“空间门槛”的前提下，才执行昨日超跌+今日放量的突破判定
    prev_rsi, prev_vol = np.roll(rsi6, 1), np.roll(vol_ratio, 1)
    sig_add = (prev_rsi <= RSI_MAX) & (prev_vol <= SHRINK_VOL_MAX) & \
              (vol_ratio >= ADD_POS_VOL_RATIO) & (change > 0) & \
              entry_confirm & basic_filter # <--- 此处补全了空间约束

    return np.select([sig_add, sig_break, sig_base_build], ["🔥放量加仓", "🚀底部放量", "💎地量筑底"], default=None)

# =====================================================================
#                          回测与主流程 (保持不变)
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
            period_low = ind['low'][idx+1 : idx+HOLD_DAYS+1].min()
            # 止损逻辑：若持仓期最低价触及-10%则止损
            if (period_low - entry_p) / entry_p <= STOP_LOSS_LIMIT:
                trades.append(STOP_LOSS_LIMIT)
            else:
                # 否则持仓至20天期满卖出
                trades.append((ind['close'][idx+HOLD_DAYS] - entry_p) / entry_p)
        return trades
    except: return None

def main():
    start_t = datetime.now()
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    name_map = {}
    if os.path.exists(NAME_MAP_FILE):
        try:
            n_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
            name_map = dict(zip(n_df['code'].str.zfill(6), n_df['name']))
        except: pass

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"🧬 启动空间约束版回测 | 样本: {len(files)}")
    
    with Pool(processes=cpu_count()) as pool:
        all_rets = [t for res in pool.map(backtest_task, files) if res for t in res]
    
    stats_msg = "数据不足"
    if all_rets:
        rets = np.array(all_rets)
        stats_msg = f"总交易: {len(rets)} | 胜率: {np.sum(rets>0)/len(rets):.2%} | 平均收益: {np.mean(rets):.2%}"

    picked = []
    print("🎯 正在扫描实战信号...")
    for f in files:
        code = os.path.basename(f)[:6]; name = name_map.get(code, "未知")
        if "ST" in name or "退" in name: continue
        try:
            ind = calculate_indicators(pd.read_csv(f))
            if ind:
                sigs = get_signals_fast(ind)
                if sigs[-1] is not None:
                    picked.append({
                        "代码": code, "名称": name, "信号": sigs[-1], 
                        "价格": ind['close'][-1], "量比": round(ind['vol_ratio'][-1], 2),
                        "振幅%": round(ind['avg_amp_3'][-1]*100, 2),
                        "空间%": round(ind['potential'][-1], 1)
                    })
        except: continue

    report_path = os.path.join(REPORT_DIR, f"Report_{datetime.now(SHANGHAI_TZ).strftime('%Y%m%d')}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 🛡️ 极致量化系统(空间限制版)\n\n日期: {datetime.now(SHANGHAI_TZ).strftime('%Y-%m-%d')}\n\n")
        f.write(f"### 🧪 策略体检看板\n> {stats_msg}\n\n")
        f.write("### 🎯 今日精选清单\n" + (pd.DataFrame(picked).to_markdown(index=False) if picked else "暂无符合25%空间要求的信号。"))
        f.write(f"\n\n---\n**系统修正**：已强制要求“放量加仓”信号也必须满足25%的空间潜力。")
    
    print(f"✅ 执行完毕 | {stats_msg} | 今日信号: {len(picked)}")

if __name__ == "__main__":
    main()
