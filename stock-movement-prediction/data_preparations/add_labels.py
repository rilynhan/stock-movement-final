import sys

import os
import datetime
import numpy as np
import pandas as pd
from lxml import etree
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline

from use_akshare import Akshare

FORWARD_DAYS = 10
THRESH_VERY = 0.5
THRESH = 0.1
# not too recent
END_DATE = "2023-08-04"

from loguru import logger

# The result_path should be the results with only titles which is the IN path
result_path = "../fin_data/results_with_content_clean/"
file_list = [os.path.join(result_path, file_name) for file_name in os.listdir(result_path)]
result_path = "../fin_data/clickhouse_data/"
file_list += [os.path.join(result_path, file_name) for file_name in os.listdir(result_path)]

# The base_path should be the results with labels which is the OUT path
base_path = "../fin_data/content_with_labels/"


# def add_label(x, df_price, foward_days = 5, threshold = 0.02, threshold_very = 0.06):
#     publish_date = x.publish_date.strftime("%Y-%m-%d")
#     last_df = df_price[df_price.time < publish_date]
#     if last_df.empty:
#         return "No data"
#     last_date = last_df.iloc[-1].name
#     this_date_index = last_date + 1
#     next_date_index = this_date_index + foward_days
    
#     if next_date_index >= df_price.shape[0]-1:
#         return "No data"
#     else:
#         this = df_price[df_price.index == this_date_index].close.values[0]
#         next_ = df_price[df_price.index == next_date_index].close.values[0]
#         change = (next_ - this)/this
#         if change > threshold_very:
#             return "very positive"
#         elif change > threshold:
#             return "positive"
#         elif change < -threshold_very:
#             return "very negative"
#         elif change < -threshold:
#             return "negative"
#         else:
#             return "neutral"
        
# def add_trend(x, df_price, foward_days = 5, threshold = 0.02, threshold_very = 0.06):
#     publish_date = x.publish_date.strftime("%Y-%m-%d")
#     last_df = df_price[df_price.time <= publish_date]
#     if last_df.empty:
#         return "No data"
#     last_date_index = last_df.iloc[-1].name
#     prev_date_index = last_date_index - foward_days
    
#     if prev_date_index < 0:
#         return "No data"
#     else:
#         prices = [df_price[df_price.index == date_index].close.values[0] for date_index in range(prev_date_index, last_date_index+1)]
#         prices = [str(round((prices[i+1] - prices[i])/prices[i]*100, 3)) for i in range(len(prices)-1)]
#         return ','.join(prices)

def add_label_slope(x, df_price, threshold = 0.1, threshold_very = 0.5):
    publish_date = x.publish_date.strftime("%Y-%m-%d")
    last_df = df_price[df_price.time <= publish_date]
    if last_df.empty:
        return "No data"
    last_date = last_df.iloc[-1].name
    this_date_index = last_date + 1

    if this_date_index >= df_price.shape[0]-1:
        return "No data"
    else:
        this = df_price[df_price.index == this_date_index].slope.values[0]
        if this > threshold_very:
            return "very positive"
        elif this > threshold:
            return "positive"
        elif this < -threshold_very:
            return "very negative"
        elif this < -threshold:
            return "negative"
        else:
            return "neutral"
        
def add_trend_slope(x, df_price, foward_days = 10):
    publish_date = x.publish_date.strftime("%Y-%m-%d")
    last_df = df_price[df_price.time <= publish_date]
    if last_df.empty:
        return "No data"
    last_date_index = last_df.iloc[-1].name
    prev_date_index = last_date_index - foward_days
    
    if prev_date_index < 0:
        return "No data"
    else:
        prices = [df_price[df_price.index == date_index].slope.values[0] for date_index in range(prev_date_index, last_date_index+1)]
        prices = [str(round((prices[i+1] - prices[i])/prices[i], 3)) for i in range(len(prices)-1)]
        return ','.join(prices)

def get_spline_slope(prices):
    x = np.arange(len(prices))
    spline = UnivariateSpline(x, prices)
    slope = [spline.derivative(n=1)(i) for i in x]
    return slope
    

@logger.catch()
def process_label(file_name, foward_days = 10, threshold = 0.1, threshold_very = 0.5):
    df = pd.read_csv(file_name)
    df["post_publish_time"] = pd.to_datetime(df["post_publish_time"])
    df["date"] = df["post_publish_time"].dt.date
    df["time"] = df["post_publish_time"].dt.time
    df["hour"] = df["post_publish_time"].dt.hour

    if 'text_type' not in df.columns:
        df['text_type'] = ['news' for i in range(len(df))]

    # start_date = df["date"].min() - datetime.timedelta(days = 25)
    # end_date = df["date"].max() + datetime.timedelta(days = 25)
    start_date = '2012-01-01'
    end_date = END_DATE
    # start_date = start_date.strftime("%Y-%m-%d")
    # end_date = end_date.strftime("%Y-%m-%d")
    code_list = [df.code_name.unique()[0]]

    as_processor = Akshare("akshare", start_date = start_date, end_date = end_date, time_interval="daily")
    as_processor.download_data(code_list, save_path = f'../fin_data/stock_data/{os.path.basename(file_name)}')
    df_price = as_processor.dataframe
    
    df_price['slope'] = get_spline_slope(df_price['close'])
    df['publish_date'] = df.apply(lambda x:x['date'] if x['hour'] <15 else x['date'] + datetime.timedelta(days = 1), axis=1)
    # df["label"] = df.apply(lambda x:add_label(x, df_price = df_price, foward_days = foward_days, threshold = threshold, threshold_very = threshold_very), axis=1)
    # df["trend_before"] = df.apply(lambda x:add_trend(x, df_price = df_price, foward_days = foward_days, threshold = threshold, threshold_very = threshold_very), axis=1)
    df["label"] = df.apply(lambda x:add_label_slope(x, df_price = df_price, threshold = threshold, threshold_very = threshold_very), axis=1)
    df["trend_before"] = df.apply(lambda x:add_trend_slope(x, df_price = df_price, foward_days = foward_days), axis=1)
    title = df['title'].tolist()
    text_type = df['text_type'].tolist()
    df['title'] = ['【公告】'+title[i] if text_type[i] == 'notice' else title[i] for i in range(len(title))]

    columns_needed = ['publish_date', 'slope', 'code_name', 'text_type', 'title', 'content', 'label', 'trend_before']
    for c in df.columns:
        if c not in columns_needed:
            df.drop(c, axis=1, inplace=True)
    return df

df_processed = pd.DataFrame({
        'publish_date': [],
        'code_name': [],
        'text_type': [],
        'title': [],
        'content': [],
        'label': []
    })
for id,file_name in enumerate(file_list):
    print(id, file_name)
    file_df = process_label(file_name, foward_days = FORWARD_DAYS, threshold = THRESH, threshold_very = THRESH_VERY)
    df_processed = df_processed.append(file_df)

out_path = os.path.join(base_path, 'all_slope.csv')
df_processed.to_csv(out_path, index = False)