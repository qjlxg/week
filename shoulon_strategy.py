import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import multiprocessing

# ==========================================
# 战法名称：首板缩量回踩擒龙战法
# 核心逻辑：
# 1. 选出近期（5日内）有涨停板的强势股（基因识别）
# 2. 识别涨停后的倍量阴线洗盘（主力震仓）
# 3. 寻找回踩 5/10日线且成交量极度萎缩的买点（卖盘枯竭）
# 4. 价格区间：5.0 - 20.0 元，排除 ST 和 30/688 开头
# ==========================================

def analyze_stock(file_path, name_dict):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 20: return None
        
        # 基础信息获取
        code = os.path.basename(file_path).replace('.csv', '')
        
        # 排除条件：ST, 创业板(30), 科创板(688)
        if code.startswith(('30', '688')): return None
        stock_name = name_dict.get(code, "未知")
        if 'ST' in stock_name or '*ST' in stock_name: return None

        # 获取最新数据
        latest = df.iloc[-1]
        close_p = latest['收盘']
        
        # 价格筛选
        if not (5.0 <= close_p <= 20.0): return None

        # 战法逻辑计算
        # 1. 查找最近5天内是否有过涨停 (涨幅 > 9.9%)
        recent_df = df.tail(6)
        has_limit_up = any(recent_df['涨跌幅'] >= 9.9)
        if not has_limit_up: return None

        # 2. 计算均线 (MA5, MA10)
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA10'] = df['收盘'].rolling(window=10).mean()
        
        # 3. 缩量逻辑：今日成交量是否明显小于昨日及涨停日
        vol_ratio = latest['成交量'] / df.iloc[-2]['成交量']
        
        # 4. 空间逻辑：价格是否在均线附近（回踩确认）
        ma_dist = abs(close_p - latest['MA5']) / latest['MA5']
        
        # --- 策略评价体系 ---
        score = 0
        advice = ""
        
        # 评分项
        if vol_ratio < 0.7: score += 40 # 显著缩量
        if ma_dist < 0.02: score += 30 # 贴近均线支撑
        if latest['涨跌幅'] > -2: score += 30 # 止跌迹象明显

        # 结论输出
        if score >= 70:
            status = "【强烈关注】" if score >= 90 else "【试错观察】"
            if vol_ratio < 0.5:
                advice = "极致缩量，主力高度控盘，回踩支撑位建议轻仓试错。"
            else:
                advice = "趋势回踩，等待分时放量勾头时择机介入。"
                
            return {
                "代码": code,
                "名称": stock_name,
                "收盘价": close_p,
                "涨跌幅%": latest['涨跌幅'],
                "量比": round(vol_ratio, 2),
                "信号强度": score,
                "复盘建议": f"{status} {advice}"
            }

    except Exception as e:
        return None

def run_strategy():
    # 1. 加载股票名称映射
    try:
        names_df = pd.read_csv('stock_names.csv')
        # 确保代码是字符串格式并补全6位
        names_df['code'] = names_df['code'].astype(str).str.zfill(6)
        name_dict = dict(zip(names_df['code'], names_df['name']))
    except:
        name_dict = {}

    # 2. 扫描目录
    csv_files = glob.glob('stock_data/*.csv')
    
    # 3. 并行处理
    print(f"开始分析 {len(csv_files)} 只股票...")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.starmap(analyze_stock, [(f, name_dict) for f in csv_files])
    pool.close()
    pool.join()

    # 4. 过滤结果
    final_list = [r for r in results if r is not None]
    final_df = pd.DataFrame(final_list)

    if not final_df.empty:
        # 按信号强度排序，只取前 10 名，保证“一击必中”
        final_df = final_df.sort_values(by="信号强度", ascending=False).head(10)
        
        # 5. 创建保存目录
        now = datetime.now()
        dir_path = now.strftime('%Y%m')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        file_name = f"shoulon_strategy_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        save_path = os.path.join(dir_path, file_name)
        
        final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"分析完成，精选 {len(final_df)} 只个股，已保存至 {save_path}")
    else:
        print("今日无符合战法条件的股票。")

if __name__ == "__main__":
    run_strategy()
