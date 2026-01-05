import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count
from datetime import datetime
import pytz

# =====================================================================
#                          核心参数区 (盈利模型参数)
# =====================================================================
# --- 大盘风控 ---
INDEX_CODE = 'sz000300'        # 判定基准：沪深300。代表市场主流价值中枢。
ENABLE_MARKET_FILTER = False   # 逻辑开关：超跌策略在弱市中往往能捕捉到极致乖离机会，故默认关闭。

# --- 选股门槛：决定安全边际 ---
MIN_PRICE = 5.0                # 股价门槛：规避面值退市风险及低价僵尸股。
MAX_AVG_TURNOVER_30 = 2.5      # 活跃度限制：寻找筹码沉淀、非散户博弈区的标的，过滤过热股。
# 空间门槛：要求现价距离MA60（60日线）至少有25%的距离，确保足够的“反弹获利跑道”。
MIN_PROFIT_POTENTIAL = 25      

# --- 技术指标阈值：决定进场深度 ---
RSI_MAX = 25                   # 强力超跌区边界：RSI6低于25意味着短期情绪跌入冰点。
KDJ_K_MAX = 25                 # KDJ超卖确认：K值低于25代表价格处于绝对低位区域。
MAX_TODAY_CHANGE = 3.0         # 涨幅限制：拒绝追高。仅限当日涨幅在3%以内的温和启动标的。

# --- 量能确认：识别主力动向 ---
SHRINK_VOL_MAX = 0.85          # 极致缩量标准：代表卖盘枯竭，地量见地价。
ADD_POS_VOL_RATIO = 1.5        # 关键放量标准：代表主力资金回补，确认筑底成功。

# --- 交易执行：核心获利与风控配置 ---
STOP_LOSS_LIMIT = -0.10        # 硬性止损：适配超跌股的大幅波动，给底部震荡留出10%空间。
HOLD_DAYS = 20                 # 最大持仓时间：若未触碰止盈位，20个交易日后强制离场。

# --- 环境配置 ---
DATA_DIR = "stock_data"
REPORT_DIR = "results"
NAME_MAP_FILE = 'stock_names.csv' 
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')

# =====================================================================
#                          指标计算引擎
# =====================================================================

def calculate_indicators(df):
    """计算核心技术指标，加入 MA20/MA60 用于实战止盈判定"""
    if df is None or len(df) < 65: return None
    
    # 确保日期升序
    d_col = next((c for c in ['日期', 'date', '时间'] if c in df.columns), None)
    if d_col: df = df.sort_values(d_col).reset_index(drop=True)
    
    try:
        close = df['收盘'].values if '收盘' in df.columns else df['close'].values
        high = df['最高'].values if '最高' in df.columns else df['high'].values
        low = df['最低'].values if '最低' in df.columns else df['low'].values
        vol = df['成交量'].values if '成交量' in df.columns else df['volume'].values
        turnover = df['换手率'].values if '换手率' in df.columns else np.zeros(len(df))
    except: return None
    
    # RSI6 计算
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0); dn = np.where(delta < 0, -delta, 0)
    def rma(x, n): return pd.Series(x).ewm(alpha=1/n, adjust=False).mean().values
    rsi6 = 100 - (100 / (1 + (rma(up, 6) / np.where(rma(dn, 6) == 0, 1e-9, rma(dn, 6)))))
    
    # KDJ (9,3,3) 计算
    low_9 = pd.Series(low).rolling(9).min().values
    high_9 = pd.Series(high).rolling(9).max().values
    rsv = (close - low_9) / np.where(high_9 - low_9 == 0, 1e-9, high_9 - low_9) * 100
    kdj_k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
    
    # 多周期均线系统
    ma5 = pd.Series(close).rolling(5).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values # 华立股份式：第一止盈观察线
    ma60 = pd.Series(close).rolling(60).mean().values # 终极回归：目标止盈线
    
    # 空间潜力判定
    potential = (ma60 - close) / np.where(close == 0, 1, close) * 100
    
    # 筑底特征：5日线斜率 + 3日振幅收敛
    ma5_change = (ma5 - np.roll(ma5, 1)) / np.where(ma5 == 0, 1, ma5)
    amplitude = (high - low) / np.where(close == 0, 1, close)
    avg_amp_3 = pd.Series(amplitude).rolling(3).mean().values
    
    # 量能判定
    vol_ma5 = pd.Series(vol).shift(1).rolling(5).mean().values
    vol_ratio = vol / np.where(vol_ma5 == 0, 1e-9, vol_ma5)
    
    # 辅助过滤：不破3日低点且计算涨跌幅
    min_3d_low = pd.Series(low).shift(1).rolling(3).min().values
    change = pd.Series(close).pct_change().values * 100
    avg_turnover_30 = pd.Series(turnover).rolling(30).mean().values

    return {
        'close': close, 'low': low, 'high': high, 'rsi6': rsi6, 'kdj_k': kdj_k,
        'ma5': ma5, 'ma20': ma20, 'ma60': ma60, 'vol_ratio': vol_ratio, 
        'ma5_change': ma5_change, 'avg_amp_3': avg_amp_3, 'min_3d_low': min_3d_low, 
        'change': change, 'avg_turnover_30': avg_turnover_30, 'potential': potential
    }

# =====================================================================
#                          信号判定核心 (保持逻辑不变)
# =====================================================================

def get_signals_fast(ind):
    """三维信号识别：放量加仓、底部突破、地量筑底"""
    close, rsi6, kdj_k = ind['close'], ind['rsi6'], ind['kdj_k']
    ma5, vol_ratio, potential = ind['ma5'], ind['vol_ratio'], ind['potential']
    ma5_c, amp3, min_3d, change, turn30 = ind['ma5_change'], ind['avg_amp_3'], ind['min_3d_low'], ind['change'], ind['avg_turnover_30']
    
    # 全局基础过滤
    basic_filter = (potential >= MIN_PROFIT_POTENTIAL) & (close >= MIN_PRICE) & \
                   (turn30 <= MAX_AVG_TURNOVER_30) & (change <= MAX_TODAY_CHANGE)
    entry_confirm = (close >= ma5) & (ind['low'] >= min_3d)

    # 1. 💎地量筑底 (均线走平，K线变小)
    sig_base_build = (rsi6 <= 30) & (ma5_c >= -0.005) & (amp3 <= 0.025) & \
                     (vol_ratio <= SHRINK_VOL_MAX) & entry_confirm & basic_filter
    # 2. 🚀底部突破 (极致超跌后的第一波反弹)
    sig_break = (rsi6 <= RSI_MAX) & (kdj_k <= KDJ_K_MAX) & (vol_ratio > SHRINK_VOL_MAX) & \
                (vol_ratio <= 2.0) & (change > 0) & entry_confirm & basic_filter
    # 3. 🔥放量加仓 (极度缩量后的强势转折)
    prev_rsi, prev_vol = np.roll(rsi6, 1), np.roll(vol_ratio, 1)
    sig_add = (prev_rsi <= RSI_MAX) & (prev_vol <= SHRINK_VOL_MAX) & \
              (vol_ratio >= ADD_POS_VOL_RATIO) & (change > 0) & entry_confirm & basic_filter

    return np.select([sig_add, sig_break, sig_base_build], ["🔥放量加仓", "🚀底部突破", "💎地量筑底"], default=None)

# =====================================================================
#                          动态回测引擎 (华立股份实战模拟)
# =====================================================================

def backtest_task(file_path):
    """
    实战仿真：
    1. 每日实时监测：若股价最高点触碰 MA60 或 摸到 MA20(且获利>5%) 立即卖出。
    2. 每日实时监测：若最低价触碰 -10% 止损位，立即离场。
    """
    try:
        df = pd.read_csv(file_path)
        ind = calculate_indicators(df)
        if ind is None: return None
        sigs = get_signals_fast(ind)
        indices = np.where(sigs != None)[0]
        
        trades = []
        for idx in indices:
            if idx + HOLD_DAYS >= len(ind['close']): continue
            
            entry_p = ind['close'][idx]  # 买入价格
            sl_price = entry_p * (1 + STOP_LOSS_LIMIT) # 预设止损价
            is_closed = False
            
            # 进入持仓期逐日监控
            for day in range(1, HOLD_DAYS + 1):
                curr_idx = idx + day
                h, l, ma20, ma60 = ind['high'][curr_idx], ind['low'][curr_idx], ind['ma20'][curr_idx], ind['ma60'][curr_idx]
                
                # A. 触发硬止损
                if l <= sl_price:
                    trades.append(STOP_LOSS_LIMIT); is_closed = True; break
                
                # B. 触发 MA60 终极止盈 (回归完成)
                if h >= ma60:
                    trades.append((ma60 - entry_p) / entry_p); is_closed = True; break
                
                # C. 触发 MA20 动态止盈 (华立股份模式：反弹遇阻)
                # 要求摸到MA20且此时已有一定获利(5%以上)，防止微利时被洗出
                if h >= ma20 and (h / entry_p - 1) >= 0.05:
                    trades.append((ma20 - entry_p) / entry_p); is_closed = True; break
            
            # D. 到期强平
            if not is_closed:
                trades.append((ind['close'][idx + HOLD_DAYS] - entry_p) / entry_p)
        return trades
    except: return None

# =====================================================================
#                          执行主流程
# =====================================================================

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # 运行回测
    print(f"🧬 正在以实战参数运行【双重均线动态止盈】回测 | 样本: {len(files)}")
    with Pool(processes=cpu_count()) as pool:
        all_rets = [t for res in pool.map(backtest_task, files) if res for t in res]
    
    stats_msg = "数据不足"
    if all_rets:
        rets = np.array(all_rets)
        stats_msg = f"总交易: {len(rets)} | 胜率: {np.sum(rets>0)/len(rets):.2%} | 平均收益: {np.mean(rets):.2%}"

    # 扫描信号
    picked = []
    for f in files:
        try:
            df = pd.read_csv(f)
            ind = calculate_indicators(df)
            if ind:
                sigs = get_signals_fast(ind)
                if sigs[-1] is not None:
                    picked.append({
                        "代码": os.path.basename(f)[:6], "信号": sigs[-1], "价格": ind['close'][-1],
                        "MA20止盈点": round(ind['ma20'][-1], 2), "MA60终极点": round(ind['ma60'][-1], 2),
                        "空间%": round(ind['potential'][-1], 1)
                    })
        except: continue

    # 生成报告
    report_path = os.path.join(REPORT_DIR, f"Live_Report_{datetime.now(SHANGHAI_TZ).strftime('%Y%m%d')}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 🛡️ 极致量化实战版报告\n\n### 🧪 策略体检看板\n> {stats_msg}\n\n")
        f.write("### 🎯 今日精选清单 (建议配合MA20/MA60条件单操作)\n" + (pd.DataFrame(picked).to_markdown(index=False) if picked else "今日无信号"))
    
    print(f"✅ 执行完毕 | {stats_msg}")

if __name__ == "__main__":
    main()
