import pandas as pd
import numpy as np
import os, glob, pytz
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

BJ_TZ = pytz.timezone('Asia/Shanghai')

def run_backtest(file_path, names_dict):
    try:
        code = os.path.basename(file_path).split('.')[0]
        # 【硬性过滤对齐】仅限深沪A股，排除 30 (创业板) 等
        if not (code.startswith('60') or code.startswith('00')):
            return []
        
        # 排除 ST
        stock_name = names_dict.get(code, "未知")
        if "ST" in stock_name: return []

        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        df.sort_values('日期', inplace=True)
        df.set_index('日期', inplace=True)
        
        # 转换为周线
        w_df = df.resample('W').agg({
            '收盘': 'last', '成交量': 'sum', '最高': 'max', '最低': 'min', '开盘': 'first'
        })
        
        if len(w_df) < 20: return []
        
        w_df['MA5'] = w_df['收盘'].rolling(5).mean()
        w_df['MA10'] = w_df['收盘'].rolling(10).mean()
        w_df['V_MA5'] = w_df['成交量'].rolling(5).mean()
        
        trades = []
        in_pos = False
        buy_p, buy_d = 0, None
        
        for i in range(15, len(w_df)-1):
            curr, prev = w_df.iloc[i], w_df.iloc[i-1]
            if not in_pos:
                # --- 筛选条件完全对齐精选脚本 ---
                # 1. 价格过滤 (5-20元)
                if not (5.0 <= curr['收盘'] <= 20.0): continue
                
                # 2. 趋势斜率：MA10 向上且 MA5 > MA10
                if curr['MA10'] <= prev['MA10'] or curr['MA5'] <= curr['MA10']: continue
                
                # 3. 量能门槛：1.5 倍以上 (零值保护)
                vol_ratio = curr['成交量'] / curr['V_MA5'] if curr['V_MA5'] > 0 else 0
                if vol_ratio < 1.5: continue
                
                # 4. 偏离门槛：3% 以内
                bias_5 = (curr['收盘'] - curr['MA5']) / curr['MA5'] if curr['MA5'] > 0 else 10
                if bias_5 > 0.03: continue
                
                # 5. 形态确认：阳线实体 (收盘 > 开盘)
                if not (curr['收盘'] > curr['开盘']): continue
                
                # 触发买入：下周一开盘买入
                in_pos = True
                buy_p = w_df.iloc[i+1]['开盘']
                buy_d = w_df.index[i+1]
                
            elif in_pos:
                # 离场逻辑：单笔止损 5% 或 MA5 死叉 MA10  
                if curr['收盘'] < buy_p * 0.95 or curr['MA5'] < curr['MA10']:
                    trades.append({
                        '年份': buy_d.year, 
                        '盈亏%': ((curr['收盘'] - buy_p) / buy_p * 100) - 0.3
                    })
                    in_pos = False
        return trades
    except:
        return []

def main():
    names_df = pd.read_csv('stock_names.csv', dtype={'code': str})
    names_dict = dict(zip(names_df['code'], names_df['name']))
    
    all_t = []
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(run_backtest, f, names_dict) for f in glob.glob('stock_data/*.csv')]
        for f in futures:
            all_t.extend(f.result())
    
    if all_t:
        df = pd.DataFrame(all_t)
        # 年度汇总
        annual = df.groupby('年份')['盈亏%'].agg([
            ('交易次数', 'count'),
            ('胜率%', lambda x: (x > 0).sum()/len(x)*100),
            ('均益%', 'mean'),
            ('盈亏比', lambda x: abs(x[x>0].mean()/x[x<0].mean()) if (x<0).any() else 0)
        ]).round(2)
        
        # 总体汇总
        summary = pd.DataFrame([{
            '年份': '所有年份总计',
            '交易次数': len(df),
            '胜率%': round((df['盈亏%'] > 0).sum()/len(df)*100, 2),
            '均益%': round(df['盈亏%'].mean(), 2),
            '盈亏比': round(abs(df[df['盈亏%'] > 0]['盈亏%'].mean()/df[df['盈亏%'] < 0]['盈亏%'].mean()) if (df['盈亏%']<0).any() else 0, 2)
        }])
        
        final_report = pd.concat([annual.reset_index(), summary], ignore_index=True)
        
        now = datetime.now(BJ_TZ)
        folder = now.strftime('%Y-%m')
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/3_年度胜率复盘报告_{now.strftime('%Y%m%d')}.csv"
        final_report.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"对齐版回测报告已生成。")

if __name__ == '__main__':
    main()
