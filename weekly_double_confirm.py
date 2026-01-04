import pandas as pd
import numpy as np
import os, glob, pytz
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

BJ_TZ = pytz.timezone('Asia/Shanghai')

def analyze_confirm_logic(file_path, names_dict):
    try:
        code = os.path.basename(file_path).split('.')[0]
        # 【硬性过滤】仅限深沪主板，排除创业板 30
        if not (code.startswith('60') or code.startswith('00')):
            return None
        
        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
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
        
        # 【硬性过滤】价格 5.0 - 20.0 元 & 排除 ST
        if not (5.0 <= curr['收盘'] <= 20.0): return None
        stock_name = names_dict.get(code, "未知")
        if "ST" in stock_name: return None
        
        # 【形态过滤】MA10上升且MA5 > MA10
        if curr['MA10'] <= prev['MA10'] or curr['MA5'] <= curr['MA10']: return None

        # 【量能与偏离】1.2倍量 & 3%偏离限制
        vol_ratio = curr['成交量'] / curr['V_MA5'] if curr['V_MA5'] > 0 else 0
        bias_5 = (curr['收盘'] - curr['MA5']) / curr['MA5'] if curr['MA5'] > 0 else 10
        if vol_ratio < 1.5 or bias_5 > 0.03: return None

        # 【阳线确认】收盘 > 开盘
        if curr['收盘'] <= curr['开盘']: return None

        history_wash = w_df.iloc[-4:-1]['成交量'].min() < curr['V_MA5'] * 0.7

        return {
            '代码': code, '名称': stock_name, '最新收盘': round(curr['收盘'], 2),
            '量能强度': round(vol_ratio, 2), '5周线偏离%': round(bias_5 * 100, 2),
            '洗盘状态': "有缩量回踩(优质)" if history_wash else "持续放量(观察)",
            '3w实战建议': "分配1.5w(狙击)" if history_wash else "分配1w(轻仓试探)"
        }
    except: return None

def main():
    names_df = pd.read_csv('stock_names.csv', dtype={'code': str})
    names_dict = dict(zip(names_df['code'], names_df['name']))
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_confirm_logic, f, names_dict) for f in glob.glob('stock_data/*.csv')]
        results = [f.result() for f in futures if f.result()]
    if results:
        res_df = pd.DataFrame(results).sort_values(by='量能强度', ascending=False)
        folder = datetime.now(BJ_TZ).strftime('%Y-%m')
        os.makedirs(folder, exist_ok=True)
        res_df.to_csv(f"{folder}/2_实战狙击单_{datetime.now(BJ_TZ).strftime('%Y%m%d')}.csv", index=False, encoding='utf-8-sig')

if __name__ == '__main__': main()
