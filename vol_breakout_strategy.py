import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from datetime import datetime

"""
战法名称：量价突破回踩战法 (Volume Expansion & Contraction Strategy)
逻辑说明：
1. 识别放量：寻找近期出现过“异常放量”的个股（单日成交量 > 前5日均量2.5倍），代表主力进场。
2. 识别缩量：放量后股价不跌破放量日开盘价，且成交量快速萎缩，代表筹码锁定。
3. 买入点：股价回踩5日均线(MA5)且成交量处于低位，或缩量后再次放量启动。
4. 排除标准：价格介于 5.0 - 20.0 元，排除ST、创业板(30)、科创板(68)、北交所。
"""

# --- 配置参数 ---
DATA_DIR = 'stock_data'
NAMES_FILE = 'stock_names.csv'
PRICE_MIN = 5.0
PRICE_MAX = 20.0

def run_backtest(df, signal_indices):
    """历史回测模块：评估该股历史上触发该战法后的表现"""
    profits = []
    for idx in signal_indices:
        # 确保有足够的后续数据计算收益 (持有5个交易日)
        if idx + 5 >= len(df): continue
        buy_price = df.loc[idx, 'close']
        max_price_5d = df.loc[idx+1 : idx+5, 'high'].max()
        profit = (max_price_5d - buy_price) / buy_price * 100
        profits.append(profit)
    
    if not profits: return 0.0, 0.0
    win_rate = len([p for p in profits if p > 3.0]) / len(profits) * 100  # 涨幅超3%视为成功
    avg_gain = np.mean(profits)
    return win_rate, avg_gain

def analyze_stock(file_path, stock_names):
    """单只股票深度扫描逻辑"""
    try:
        df = pd.read_csv(file_path)
        if len(df) < 40: return None
        
        # 字段映射
        df.columns = ['date', 'code', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'pct_val', 'turnover']
        
        # 1. 基础筛选 (排除ST、创业板、科创板、北交所及价格区间)
        code = str(df['code'].iloc[-1]).zfill(6)
        name = stock_names.get(code, "未知")
        if code.startswith(('30', '68', '8', '4')) or 'ST' in name: return None
        
        last_close = df['close'].iloc[-1]
        if not (PRICE_MIN <= last_close <= PRICE_MAX): return None

        # 2. 技术指标计算
        df['ma5'] = df['close'].rolling(5).mean()
        df['vol_ma5'] = df['volume'].rolling(5).mean().shift(1)
        
        # 3. 历史信号回测 (过去250个交易日)
        # 逻辑：成交量 > 5日均量2.5倍 且 涨幅 > 2.5%
        all_signals = df[(df['volume'] > df['vol_ma5'] * 2.5) & (df['pct_chg'] > 2.5)].index.tolist()
        win_rate, avg_gain = run_backtest(df, all_signals[:-1]) # 排除当日信号

        # 4. 实时战法识别 (寻找最近10天内的放量基准日)
        recent = df.tail(10)
        breakout_days = recent[(recent['volume'] > recent['vol_ma5'] * 2.5) & (recent['pct_chg'] > 2.5)]
        
        if breakout_days.empty: return None
        
        # 获取最近的一个放量突破日
        v_idx = breakout_days.index[-1]
        v_day = df.loc[v_idx]
        post_break = df.loc[v_idx + 1:]
        
        score = 0
        advice = ""
        
        if post_break.empty:
            # 今日刚放量
            score = 65
            advice = "今日首发异动，放量明显。建议明日观察是否缩量，切勿盲目追高。"
        else:
            # 缩量与承接检查
            vol_is_shrinking = df['volume'].iloc[-1] < v_day['volume'] * 0.55  # 成交量萎缩至放量日的一半以下
            price_holds = df['close'].min() > v_day['open']                 # 股价未跌破放量日起涨点
            near_ma5 = last_close <= df['ma5'].iloc[-1] * 1.015               # 股价回踩至MA5附近(1.5%以内)
            
            if vol_is_shrinking: score += 35
            if price_holds: score += 25
            if near_ma5: score += 25
            if win_rate > 60: score += 15  # 历史高胜率加分

            # 自动化建议分级
            if score >= 85:
                advice = f"【一击必中】历史胜率{win_rate:.0f}%。缩量回踩MA5完美，主力高度锁仓，建议重仓介入。"
            elif score >= 70:
                advice = f"【试错观察】历史胜率{win_rate:.0f}%。量能萎缩达标，回踩支撑有效，可分批进场。"
            else:
                return None # 优中选优，低于70分舍弃

        return {
            '代码': code,
            '名称': name,
            '现价': last_close,
            '今日涨幅%': df['pct_chg'].iloc[-1],
            '信号强度': score,
            '历史胜率%': f"{win_rate:.1f}%",
            '操作建议': advice,
            '放量基准日': v_day['date']
        }
    except:
        return None

def main():
    # 加载股票名称对照表
    try:
        n_df = pd.read_csv(NAMES_FILE)
        n_df['code'] = n_df['code'].astype(str).str.zfill(6)
        stock_names = dict(zip(n_df['code'], n_df['name']))
    except:
        stock_names = {}

    # 并行扫描数据目录下的所有CSV
    csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    print(f"开始并行扫描 {len(csv_files)} 个标的...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(analyze_stock, [(f, stock_names) for f in csv_files])
    
    # 筛选有效结果
    valid_list = [r for r in results if r is not None]
    if not valid_list:
        print("今日复盘结束：未发现符合战法的高价值标的。")
        return

    # 优中选优：按强度排序并只保留前5只
    res_df = pd.DataFrame(valid_list).sort_values(by='信号强度', ascending=False).head(5)

    # 保存结果到年月目录
    now = datetime.now()
    month_dir = now.strftime('%Y%m')
    if not os.path.exists(month_dir): os.makedirs(month_dir)
    
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    filename = f"vol_breakout_strategy_{timestamp}.csv"
    save_path = os.path.join(month_dir, filename)
    
    res_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"复盘报告已生成：{save_path}")

if __name__ == "__main__":
    main()
