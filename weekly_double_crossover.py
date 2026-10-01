import pandas as pd
import numpy as np
import os, glob, pytz
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

BJ_TZ = pytz.timezone('Asia/Shanghai')

def analyze_crossover_logic(file_path, names_dict):
    try:
        code = os.path.basename(file_path).split('.')[0]
        # 【硬性过滤】仅限深沪主板，排除30创业板、688科创板、北交所
        if not (code.startswith('60') or code.startswith('00')):
            return None
        
        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        df.sort_values('日期', inplace=True)
        df.set_index('日期', inplace=True)
        
        # 【周线转换】
        w_df = df.resample('W').agg({
            '收盘': 'last', '成交量': 'sum', '最高': 'max', '最低': 'min', '开盘': 'first'
        })
        
        if len(w_df) < 20: return None
        
        w_df['MA5'] = w_df['收盘'].rolling(5).mean()
        w_df['MA10'] = w_df['收盘'].rolling(10).mean()
        w_df['V_MA5'] = w_df['成交量'].rolling(5).mean()
        
        curr, prev = w_df.iloc[-1], w_df.iloc[-2]
        
        # 【硬性过滤】价格 5.0 - 20.0 元
        if not (5.0 <= curr['收盘'] <= 20.0): return None
        
        # 【硬性过滤】排除 ST
        stock_name = names_dict.get(code, "未知")
        if "ST" in stock_name: return None

        # 【趋势形态】MA10上升且MA5 > MA10
        if curr['MA10'] <= prev['MA10'] or curr['MA5'] <= curr['MA10']:
            return None

        vol_ratio = curr['成交量'] / curr['V_MA5'] if curr['V_MA5'] > 0 else 0
        if vol_ratio < 0.8: return None

        bias_5 = (curr['收盘'] - curr['MA5']) / curr['MA5'] if curr['MA5'] > 0 else 10
        if bias_5 > 0.05: return None

        has_wash = (w_df.iloc[-5:-1]['成交量'] < w_df.iloc[-5:-1]['V_MA5'] * 0.7).any()

        return {
            '代码': code, '名称': stock_name, '收盘价': round(curr['收盘'], 2),
            '量能倍数': round(vol_ratio, 2), '5周偏离%': round(bias_5 * 100, 2),
            '洗盘痕迹': "有" if has_wash else "无", '状态': "形态已成" if vol_ratio >= 1.0 else "潜伏中"
        }
    except: return None

def main():
    names_df = pd.read_csv('stock_names.csv', dtype={'code': str})
    names_dict = dict(zip(names_df['code'], names_df['name']))
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_crossover_logic, f, names_dict) for f in glob.glob('stock_data/*.csv')]
        results = [f.result() for f in futures if f.result()]
    if results:
        res_df = pd.DataFrame(results).sort_values(by='量能倍数', ascending=False)
        folder = datetime.now(BJ_TZ).strftime('%Y-%m')
        os.makedirs(folder, exist_ok=True)
        res_df.to_csv(f"{folder}/1_海选潜力池_{datetime.now(BJ_TZ).strftime('%Y%m%d')}.csv", index=False, encoding='utf-8-sig')

if __name__ == '__main__': main()
