import pandas as pd
import numpy as np
import os
from datetime import datetime
import multiprocessing as mp

"""
战法名称：量价突破回踩战法 (Volume Expansion & Contraction Strategy)
逻辑说明：
1. 识别放量：寻找近期出现过“异常放量”的个股（单日成交量 > 前5日均量2.5倍），代表主力进场。
2. 识别缩量：放量后股价不跌破放量日开盘价，且成交量快速萎缩，代表筹码锁定。
3. 买入点：股价回踩5日均线(MA5)且成交量处于低位，或缩量后再次放量启动。
4. 排除标准：价格 < 5.0 或 > 20.0，排除ST、创业板(30开头)、北交所(8/4开头)。
"""

# 配置参数
DATA_DIR = 'stock_data'
NAMES_FILE = 'stock_names.csv'
MIN_PRICE = 5.0
MAX_PRICE = 20.0

def analyze_stock(file_path, stock_names):
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 20:
            return None
        
        # 统一列名（根据用户提供格式：日期 股票代码 开盘 收盘 最高 最低 成交量...）
        df.columns = ['date', 'code', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'pct_val', 'turnover']
        
        # 基础筛选：排除ST和创业板
        code = str(df['code'].iloc[-1]).zfill(6)
        if 'ST' in stock_names.get(code, '') or code.startswith('300') or code.startswith('688') or code.startswith('8') or code.startswith('4'):
            return None
        
        # 价格筛选
        curr_price = df['close'].iloc[-1]
        if curr_price < MIN_PRICE or curr_price > MAX_PRICE:
            return None

        # 技术指标计算
        df['ma5'] = df['close'].rolling(5).mean()
        df['vol_ma5'] = df['volume'].rolling(5).mean().shift(1)
        
        # 逻辑：寻找最近10天内的放量日
        # 定义放量：当日成交量 > 前5日均量 2.5倍 且 涨幅 > 3%
        df['is_breakout'] = (df['volume'] > df['vol_ma5'] * 2.5) & (df['pct_chg'] > 3)
        
        recent_window = df.tail(10).copy()
        if not recent_window['is_breakout'].any():
            return None
            
        # 找到最近的一个放量日
        breakout_idx = recent_window[recent_window['is_breakout']].index[-1]
        breakout_price = df.loc[breakout_idx, 'close']
        
        # 缩量逻辑：放量日之后，成交量持续萎缩，且价格未大幅跌破放量日中线
        post_breakout = df.loc[breakout_idx + 1:]
        if post_breakout.empty: # 如果今天就是放量日
            strength = 60
            advice = "今日初次放量，建议观察，不宜直接追高"
        else:
            vol_contract = post_breakout['volume'].iloc[-1] < df.loc[breakout_idx, 'volume'] * 0.6
            price_stable = post_breakout['close'].min() > df.loc[breakout_idx, 'open']
            
            # 评分系统
            score = 0
            if vol_contract: score += 40  # 缩量加分
            if price_stable: score += 30  # 价格稳健加分
            if curr_price >= df['ma5'].iloc[-1] * 0.98: score += 20 # 靠近MA5加分
            
            if score >= 70:
                strength = score
                advice = "缩量回踩完成，靠近MA5，一击必中潜力股" if curr_price <= df['ma5'].iloc[-1] * 1.02 else "缩量回调中，等待回踩MA5"
            else:
                return None # 评分不高则舍弃

        return {
            '代码': code,
            '名称': stock_names.get(code, '未知'),
            '当前价': curr_price,
            '涨跌幅': df['pct_chg'].iloc[-1],
            '信号强度': strength,
            '操作建议': advice,
            '放量日': df.loc[breakout_idx, 'date']
        }

    except Exception as e:
        return None

def main():
    # 加载股票名称对照表
    try:
        names_df = pd.read_csv(NAMES_FILE)
        # 强制转换code为6位字符串
        names_df['code'] = names_df['code'].astype(str).str.zfill(6)
        stock_names = dict(zip(names_df['code'], names_df['name']))
    except:
        stock_names = {}

    # 获取所有CSV文件路径
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    # 并行处理
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(analyze_stock, [(f, stock_names) for f in files])
    pool.close()
    pool.join()
    
    # 过滤无效结果并排序
    valid_results = [r for r in results if r is not None]
    final_df = pd.DataFrame(valid_results)
    
    if not final_df.empty:
        final_df = final_df.sort_values(by='信号强度', ascending=False).head(5) # 只取最强的5只

    # 创建保存目录
    now = datetime.now()
    dir_name = now.strftime('%Y%m')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # 保存结果
    file_name = f"vol_breakout_strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    save_path = os.path.join(dir_name, file_name)
    final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"扫描完毕，结果已保存至: {save_path}")

if __name__ == "__main__":
    main()
