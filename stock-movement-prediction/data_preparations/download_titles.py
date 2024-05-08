import sys
sys.path.append("../")

import os
import multiprocessing as mp
import pandas as pd
from finnlp.data_sources.news.eastmoney_streaming import Eastmoney_Streaming  # https://github.com/AI4Finance-Foundation/FinNLP

df = pd.read_csv("hs_restricted.csv")
stock_list = df.SECURITY_CODE.unique()
stock_list = [str(s).zfill(6) for s in stock_list]

# ATTENTION! Should replace this with your results path!
al_re = os.listdir(r"../fin_data/scrape/results")
al_re = [al.split(".")[0] for al in al_re]

def get_news_data( stock ):
    print(f"Collecting {stock}")
    if stock in al_re:
        return
    
    # Detailed configs can be found here: https://www.kuaidaili.com/usercenter/tps/
    config = {
        "use_proxy": "kuaidaili",
        "max_retry": 5,
        # "proxy_pages": 5,
        "tunnel": 'r354.kdltps.com:15818',
        "username": 't18923049319924',
        "password": 'x464z3b7',
    }
    
    # ATTENTION! Should replace this with your results path!
    result_path = r"../fin_data/scrape/results"
    result_path = os.path.join(result_path, f"{stock}.csv")
    downloader = Eastmoney_Streaming(config)
    downloader.download_streaming_stock(stock, rounds = 0)
    downloader.dataframe.to_csv(result_path, index = False)
    return downloader.dataframe


if __name__ == "__main__":
    pool_list = []
    res_list = []
    pool = mp.Pool(processes = 5)

    for i in stock_list:
        res = pool.apply_async(get_news_data, args = (i,), error_callback = lambda x:print(x))
        pool_list.append(res)

    pool.close()
    pool.join()

    for i in pool_list:
        res_list.append(i.get())

    print("All Done!")

