import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# 【战法名称：芙蓉出水 - 一阳穿四线强攻】
# 逻辑：股价在均线纠缠后，以一根放量大阳线同时突破5/10/20/60日四条均线，象征主力洗盘结束，开始主升。

def analyze_stock(file_path, name_map):
    try:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 60: return None
        
        # 统一列名处理 (匹配您的格式: 日期, 股票代码, 开盘, 收盘, 最高, 最低, 成交量, 涨跌幅, 换手率)
        last_row = df.iloc[-1]
        code = str(last_row['股票代码']).zfill(6)
        
        # --- 基础过滤 ---
        # 1. 排除ST和创业板(30)
        if 'ST' in name_map.get(code, '') or code.startswith('30'): return None
        # 2. 价格区间 (5.0 - 20.0)
        price = last_row['收盘']
        if not (5.0 <= price <= 20.0): return None
        
        # --- 计算技术指标 ---
        df['ma5'] = df['收盘'].rolling(5).mean()
        df['ma10'] = df['收盘'].rolling(10).mean()
        df['ma20'] = df['收盘'].rolling(20).mean()
        df['ma60'] = df['收盘'].rolling(60).mean()
        df['v_ma5'] = df['成交量'].rolling(5).mean()
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- 战法核心逻辑判断 ---
        # 1. 一阳穿四线：开盘价低于所有均线，收盘价高于所有均线 (或实体穿过)
        mas = [curr['ma5'], curr['ma10'], curr['ma20'], curr['ma60']]
        is_cross = curr['收盘'] > max(mas) and curr['开盘'] < min(mas[:3]) # 至少穿过短中期三线
        
        # 2. 涨幅及形态
        is_strong = curr['涨跌幅'] > 4.5  # 至少是大阳线
        is_limit = curr['涨跌幅'] >= 9.8  # 是否涨停
        
        # 3. 量能确认
        volume_ratio = curr['成交量'] / curr['v_ma5']
        is_vol_ok = volume_ratio > 1.5 # 放量1.5倍以上
        
        if is_cross and is_strong and is_vol_ok:
            # --- 智能评分系统 ---
            score = 60
            if is_limit: score += 20
            if 3 <= curr['换手率'] <= 10: score += 10 # 换手率健康
            if curr['收盘'] > curr['ma60']: score += 10 # 站稳长趋势线
            
            # 操作建议
            if score >= 90:
                advice = "【极强：重仓关注】一箭穿心涨停，主力高度控盘，次日择机介入。"
            elif score >= 75:
                advice = "【走强：试错介入】形态标准且放量，建议底仓试错。"
            else:
                advice = "【观察：暂缓动手】形态尚可但强度略逊，等待回踩确认。"
                
            return {
                "代码": code,
                "名称": name_map.get(code, "未知"),
                "收盘价": price,
                "涨跌幅": f"{curr['涨跌幅']}%",
                "换手率": f"{curr['换手率']}%",
                "量比": round(volume_ratio, 2),
                "评分": score,
                "操作建议": advice
            }
    except Exception as e:
        return None
    return None

def main():
    # 加载股票名称映射
    name_df = pd.read_csv('stock_names.csv')
    name_map = dict(zip(name_df['code'].astype(str).str.zfill(6), name_df['name']))
    
    # 扫描数据文件
    files = glob.glob('stock_data/*.csv')
    
    # 并行处理提高速度
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_stock, f, name_map) for f in files]
        for f in futures:
            res = f.result()
            if res: results.append(res)
            
    # 输出结果
    if results:
        final_df = pd.DataFrame(results).sort_values(by="评分", ascending=False)
        
        # 创建年月目录
        now = datetime.now()
        dir_path = now.strftime("%Y-%m")
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        
        filename = f"{dir_path}/furong_chushui_strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        final_df.to_csv(filename, index=False, encoding='utf_8_sig')
        print(f"复盘完成，选出 {len(final_df)} 只潜力股，结果已保存至 {filename}")
    else:
        print("今日未筛选出符合芙蓉出水战法的个股。")

if __name__ == "__main__":
    main()
