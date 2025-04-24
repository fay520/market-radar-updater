#!/usr/bin/env python3
# automated_data_collector.py
# 自动化数据收集脚本，从免费金融API获取最新数据并更新静态网站

import os
import sys
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 设置工作目录
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data' / 'daily'
HISTORY_DIR = BASE_DIR / 'data' / 'history'
FORECAST_DIR = BASE_DIR / 'forecast'
STATIC_SITE_DIR = BASE_DIR / 'static_site'
ASSETS_DIR = STATIC_SITE_DIR / 'assets'

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(STATIC_SITE_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# 当前日期
TODAY = datetime.datetime.now().strftime('%Y-%m-%d')
YESTERDAY = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

# 免费金融API配置
# Alpha Vantage API提供免费访问，但有请求限制
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')  # 使用环境变量或默认值
YAHOO_FINANCE_BASE_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/'

# 要跟踪的资产
ASSETS = {
    'DX-Y.NYB': {'name': '美元指数', 'type': 'currency'},
    '^IXIC': {'name': '纳斯达克', 'type': 'index'},
    '^GSPC': {'name': '标普500', 'type': 'index'},
    '^TNX': {'name': '10年期美债收益率', 'type': 'bond', 'multiplier': 0.1},  # 需要乘以0.1转换为百分比
    '^TYX': {'name': '30年期美债收益率', 'type': 'bond', 'multiplier': 0.1},  # 需要乘以0.1转换为百分比
    'GC=F': {'name': '黄金', 'type': 'commodity'}
}

# 科技股
TECH_STOCKS = {
    'AAPL': {'name': '苹果', 'type': 'stock'},
    'MSFT': {'name': '微软', 'type': 'stock'},
    'GOOGL': {'name': '谷歌', 'type': 'stock'},
    'AMZN': {'name': '亚马逊', 'type': 'stock'},
    'META': {'name': 'Meta', 'type': 'stock'},
    'NVDA': {'name': '英伟达', 'type': 'stock'},
    'TSLA': {'name': '特斯拉', 'type': 'stock'}
}

# 合并所有要跟踪的资产
ALL_ASSETS = {**ASSETS, **TECH_STOCKS}

def get_yahoo_finance_data(symbol, period='1mo', interval='1d'):
    """从Yahoo Finance API获取资产数据"""
    url = f"{YAHOO_FINANCE_BASE_URL}{symbol}"
    params = {
        'range': period,
        'interval': interval,
        'includePrePost': 'false',
        'events': 'div,split,earn',
        'lang': 'en-US',
        'region': 'US'
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # 检查是否有错误
        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            print(f"获取{symbol}数据失败: {data}")
            return None
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        # 创建数据框
        df = pd.DataFrame({
            'timestamp': [datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps],
            'open': quotes['open'],
            'high': quotes['high'],
            'low': quotes['low'],
            'close': quotes['close'],
            'volume': quotes['volume']
        })
        
        # 处理可能的None值
        df = df.fillna(method='ffill')
        
        # 如果是债券，需要乘以转换因子
        if symbol in ASSETS and 'multiplier' in ASSETS[symbol]:
            multiplier = ASSETS[symbol]['multiplier']
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] * multiplier
        
        return df
    
    except Exception as e:
        print(f"获取{symbol}数据时出错: {e}")
        return None

def collect_all_asset_data():
    """收集所有资产的最新数据"""
    all_data = {}
    
    for symbol, info in ALL_ASSETS.items():
        print(f"获取{info['name']}({symbol})数据...")
        df = get_yahoo_finance_data(symbol, period='1mo', interval='1d')
        
        if df is not None:
            # 保存到每日数据目录
            daily_file = DATA_DIR / f"{symbol.replace('^', '').replace('=', '_')}_{TODAY}.csv"
            df.to_csv(daily_file, index=False)
            
            # 添加到历史数据
            history_file = HISTORY_DIR / f"{symbol.replace('^', '').replace('=', '_')}_history.csv"
            
            if os.path.exists(history_file):
                history_df = pd.read_csv(history_file)
                # 删除可能重复的日期
                history_df = history_df[~history_df['timestamp'].isin(df['timestamp'])]
                # 合并新数据
                combined_df = pd.concat([history_df, df], ignore_index=True)
                # 按日期排序
                combined_df = combined_df.sort_values('timestamp')
                combined_df.to_csv(history_file, index=False)
            else:
                df.to_csv(history_file, index=False)
            
            all_data[symbol] = df
            print(f"成功获取{info['name']}数据，共{len(df)}条记录")
        else:
            print(f"获取{info['name']}数据失败")
        
        # 避免API请求限制
        time.sleep(1)
    
    return all_data

def generate_correlation_analysis(all_data):
    """生成资产间的相关性分析"""
    # 提取主要资产的收盘价
    close_prices = {}
    for symbol, info in ASSETS.items():
        if symbol in all_data and all_data[symbol] is not None:
            df = all_data[symbol]
            close_prices[info['name']] = df['close'].values
    
    # 创建相关性数据框
    if close_prices:
        correlation_df = pd.DataFrame(close_prices)
        correlation_matrix = correlation_df.corr()
        
        # 保存相关性矩阵
        correlation_matrix.to_csv(DATA_DIR / f"correlation_matrix_{TODAY}.csv")
        
        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f'资产相关性热力图 ({TODAY})')
        plt.tight_layout()
        plt.savefig(ASSETS_DIR / 'correlation_heatmap.png', dpi=300)
        plt.close()
        
        return correlation_matrix
    
    return None

def generate_market_overview(all_data):
    """生成市场概览图表"""
    # 提取主要资产的收盘价并标准化为基准100
    normalized_prices = {}
    start_dates = {}
    
    for symbol, info in ASSETS.items():
        if symbol in all_data and all_data[symbol] is not None:
            df = all_data[symbol]
            if len(df) > 0:
                start_dates[symbol] = df['timestamp'].iloc[0]
                base_price = df['close'].iloc[0]
                normalized_prices[info['name']] = [price / base_price * 100 for price in df['close']]
    
    # 找到共同的起始日期
    if normalized_prices:
        # 创建市场概览图
        plt.figure(figsize=(12, 6))
        
        for name, prices in normalized_prices.items():
            plt.plot(range(len(prices)), prices, label=name)
        
        plt.title(f'市场走势对比 (基准=100, {TODAY})')
        plt.xlabel('交易日')
        plt.ylabel('标准化价格 (基准=100)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(ASSETS_DIR / 'overall_market_changes.png', dpi=300)
        plt.close()
        
        # 计算波动率
        volatility = {}
        for symbol, info in ASSETS.items():
            if symbol in all_data and all_data[symbol] is not None:
                df = all_data[symbol]
                if len(df) > 1:
                    # 计算每日回报率
                    returns = df['close'].pct_change().dropna()
                    # 计算波动率（标准差）
                    volatility[info['name']] = returns.std() * 100  # 转换为百分比
        
        # 创建波动率图表
        if volatility:
            plt.figure(figsize=(10, 6))
            names = list(volatility.keys())
            values = list(volatility.values())
            
            # 按波动率排序
            sorted_indices = np.argsort(values)
            sorted_names = [names[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            
            bars = plt.barh(sorted_names, sorted_values, color='skyblue')
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                         f'{sorted_values[i]:.2f}%', va='center')
            
            plt.title(f'市场波动率 ({TODAY})')
            plt.xlabel('波动率 (%)')
            plt.tight_layout()
            plt.savefig(ASSETS_DIR / 'market_volatility.png', dpi=300)
            plt.close()
        
        return normalized_prices, volatility
    
    return None, None

def forecast_asset_prices(all_data, forecast_days=180):
    """预测资产未来6个月的价格走势"""
    forecasts = {}
    
    for symbol, info in ASSETS.items():
        if symbol in all_data and all_data[symbol] is not None:
            df = all_data[symbol]
            if len(df) > 30:  # 确保有足够的历史数据
                try:
                    # 提取收盘价
                    prices = df['close'].values
                    dates = df['timestamp'].values
                    
                    # 创建时间序列索引
                    ts_data = pd.Series(prices, index=pd.DatetimeIndex(dates))
                    
                    # 使用ARIMA模型进行预测
                    model = ARIMA(ts_data, order=(5,1,0))
                    model_fit = model.fit()
                    
                    # 预测未来6个月
                    forecast = model_fit.forecast(steps=forecast_days)
                    
                    # 生成未来日期
                    last_date = pd.to_datetime(dates[-1])
                    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]
                    future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
                    
                    # 保存预测结果
                    forecast_df = pd.DataFrame({
                        'date': future_dates_str,
                        'forecast': forecast.values
                    })
                    forecast_df.to_csv(FORECAST_DIR / f"{symbol.replace('^', '').replace('=', '_')}_forecast_{TODAY}.csv", index=False)
                    
                    # 绘制预测图表
                    plt.figure(figsize=(12, 6))
                    
                    # 绘制历史数据
                    plt.plot(range(len(prices)), prices, label='历史数据', color='blue')
                    
                    # 绘制预测数据
                    plt.plot(range(len(prices), len(prices) + len(forecast)), forecast, label='预测', color='red', linestyle='--')
                    
                    # 添加置信区间
                    lower_bound = forecast * 0.9
                    upper_bound = forecast * 1.1
                    plt.fill_between(range(len(prices), len(prices) + len(forecast)), 
                                    lower_bound, upper_bound, color='red', alpha=0.2)
                    
                    plt.title(f'{info["name"]}价格预测 (未来6个月)')
                    plt.xlabel('交易日')
                    plt.ylabel('价格')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(ASSETS_DIR / f'{info["name"]}_forecast.png', dpi=300)
                    plt.close()
                    
                    # 存储预测结果
                    forecasts[symbol] = {
                        'name': info['name'],
                        'last_price': prices[-1],
                        'forecast': forecast.values,
                        'forecast_dates': future_dates_str
                    }
                    
                    print(f"成功预测{info['name']}未来6个月价格")
                
                except Exception as e:
                    print(f"预测{info['name']}价格时出错: {e}")
    
    # 生成预测报告
    if forecasts:
        report = "# 市场预测报告\n\n"
        report += f"生成日期: {TODAY}\n\n"
        
        report += "## 未来6个月预测摘要\n\n"
        
        for symbol, data in forecasts.items():
            name = data['name']
            last_price = data['last_price']
            forecast_end = data['forecast'][-1]
            change_pct = (forecast_end - last_price) / last_price * 100
            
            report += f"### {name}\n\n"
            report += f"- 当前价格: {last_price:.2f}\n"
            report += f"- 6个月后预测价格: {forecast_end:.2f}\n"
            report += f"- 预计变化: {change_pct:.2f}%\n"
            
            # 添加关键转折点
            forecast = data['forecast']
            dates = data['forecast_dates']
            
            # 寻找显著变化点
            significant_changes = []
            for i in range(5, len(forecast)):
                # 计算5天移动平均线的斜率变化
                prev_slope = (forecast[i-1] - forecast[i-5]) / 5
                next_slope = (forecast[i+4] - forecast[i]) / 5 if i+4 < len(forecast) else (forecast[-1] - forecast[i]) / (len(forecast) - i - 1)
                
                # 如果斜率变化超过阈值，认为是转折点
                if abs(next_slope - prev_slope) > abs(prev_slope * 0.5):
                    significant_changes.append((i, dates[i], forecast[i], "上升" if next_slope > prev_slope else "下降"))
            
            if significant_changes:
                report += "\n关键转折点:\n\n"
                for idx, date, price, direction in significant_changes[:3]:  # 最多显示3个转折点
                    report += f"- {date}: 预计{direction}至 {price:.2f}\n"
            
            report += "\n"
        
        # 保存预测报告
        with open(FORECAST_DIR / f"forecast_report_{TODAY}.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 复制到静态网站目录
        with open(ASSETS_DIR / "forecast_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
    
    return forecasts

def update_static_website(all_data, correlation_matrix, forecasts):
    """更新静态网站内容"""
    # 更新index.html中的数据
    index_path = STATIC_SITE_DIR / 'index.html'
    
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 更新最后更新时间
        html_content = html_content.replace('数据更新时间：2025年4月24日', f'数据更新时间：{TODAY}')
        
        # 更新资产价格数据
        for symbol, info in ASSETS.items():
            if symbol in all_data and all_data[symbol] is not None:
                df = all_data[symbol]
                if len(df) > 0:
                    last_price = df['close'].iloc[-1]
                    first_price = df['close'].iloc[0]
                    change_pct = (last_price - first_price) / first_price * 100
                    
                    # 更新价格和变化百分比
                    if info['name'] == '美元指数':
                        html_content = html_content.replace('<div class="text-3xl font-bold mb-2">98.3</div>', 
                                                          f'<div class="text-3xl font-bold mb-2">{last_price:.1f}</div>')
                        html_content = html_content.replace('<span class="text-red-500 font-bold">-5.2%</span>', 
                                                          f'<span class="text-{"red" if change_pct < 0 else "green"}-500 font-bold">{change_pct:.1f}%</span>')
                    
                    elif info['name'] == '纳斯达克':
                        html_content = html_content.replace('<div class="text-3xl font-bold mb-2">16,742</div>', 
                                                          f'<div class="text-3xl font-bold mb-2">{int(last_price):,}</div>')
                        html_content = html_content.replace('<span class="text-red-500 font-bold">-8.1%</span>', 
                                                          f'<span class="text-{"red" if change_pct < 0 else "green"}-500 font-bold">{change_pct:.1f}%</span>')
                    
                    elif info['name'] == '10年期美债收益率':
                        html_content = html_content.replace('<div class="text-3xl font-bold mb-2">4.5%</div>', 
                                                          f'<div class="text-3xl font-bold mb-2">{last_price:.1f}%</div>')
                        html_content = html_content.replace('<span class="text-green-500 font-bold">+15.4%</span>', 
                                                          f'<span class="text-{"red" if change_pct < 0 else "green"}-500 font-bold">{change_pct:.1f}%</span>')
                    
                    elif info['name'] == '黄金':
                        html_content = html_content.replace('<div class="text-3xl font-bold mb-2">$3,352</div>', 
                                                          f'<div class="text-3xl font-bold mb-2">${int(last_price):,}</div>')
                        html_content = html_content.replace('<span class="text-green-500 font-bold">+10.5%</span>', 
                                                          f'<span class="text-{"red" if change_pct < 0 else "green"}-500 font-bold">{change_pct:.1f}%</span>')
        
        # 更新预测数据
        if forecasts:
            for symbol, data in forecasts.items():
                name = data['name']
                last_price = data['last_price']
                forecast_end = data['forecast'][-1]
                
                if name == '美元指数':
                    html_content = html_content.replace('预计下跌至79.4', f'预计{"下跌" if forecast_end < last_price else "上涨"}至 {forecast_end:.1f}')
                
                elif name == '纳斯达克':
                    html_content = html_content.replace('预计下跌至11,700', f'预计{"下跌" if forecast_end < last_price else "上涨"}至 {int(forecast_end):,}')
                
                elif name == '10年期美债收益率':
                    html_content = html_content.replace('预计上升至5.0', f'预计{"上升" if forecast_end > last_price else "下降"}至 {forecast_end:.1f}')
                
                elif name == '30年期美债收益率':
                    html_content = html_content.replace('预计上升至5.2', f'预计{"上升" if forecast_end > last_price else "下降"}至 {forecast_end:.1f}')
                
                elif name == '黄金':
                    html_content = html_content.replace('预计上涨至$4,397', f'预计{"上涨" if forecast_end > last_price else "下跌"}至 ${int(forecast_end):,}')
        
        # 保存更新后的HTML
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"静态网站已更新，最后更新时间: {TODAY}")

def main():
    """主函数"""
    print(f"开始执行自动数据收集和网站更新 ({TODAY})")
    
    # 收集所有资产数据
    all_data = collect_all_asset_data()
    
    if all_data:
        # 生成相关性分析
        correlation_matrix = generate_correlation_analysis(all_data)
        
        # 生成市场概览
        normalized_prices, volatility = generate_market_overview(all_data)
        
        # 预测资产价格
        forecasts = forecast_asset_prices(all_data, forecast_days=180)
        
        # 更新静态网站
        update_static_website(all_data, correlation_matrix, forecasts)
        
        print("数据收集和网站更新完成")
    else:
        print("数据收集失败，无法更新网站")

if __name__ == "__main__":
    main()
