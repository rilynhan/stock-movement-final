# from my_clickhouse.local_doc_qa import LocalDocQA
import os
import pandas as pd
import clickhouse_connect
# from clickhouse_driver import Client

code_df = pd.read_csv('hs_restricted.csv')
stock_names = code_df['SECURITY_NAME_ABBR'].tolist()
stock_codes = code_df['SECUCODE'].tolist()
stock_codes = [c.lower() for c in stock_codes]
name2code = {stock_names[i]:stock_codes[i] for i in range(len(stock_names))}

def load_notice(config):
    for stock_name in stock_names:
        query_str = f"select publishtime, title, source_url, content, file_path, stock_code from app_notice \
            where content like \'%{stock_name}%\' and title is not null"
            # and (length(content) - length(replace(content, \'{stock_name}\', \'\'))) / length(\'{stock_name}\') >= 1   
            # and length(content) < 5000 \
            # limit 10
        # result_list = client.execute(query_str)
        client = clickhouse_connect.get_client(host=config['host'], username=config['user'], password=config['password'], database=config['database'])
        result_list = client.query(query_str).result_set
        result_dict = {
            'post_publish_time': [], # in keeping with FinNLP
            'title': [],
            'source_url': [],
            'content': [],
            'file_path': []
        }
        for res in result_list:
            content = res[3]
            if stock_name not in content[:2000]:   
                continue
            result_dict['post_publish_time'].append(res[0])
            result_dict['title'].append(res[1])
            result_dict['source_url'].append(res[2])
            result_dict['content'].append(res[3])
            result_dict['file_path'].append(res[4])
        result_dict['code_name'] = [name2code[stock_name] for i in range(len(result_dict['title']))]
        result_dict['text_type'] = ['notice' for i in range(len(result_dict['title']))]
        df = pd.DataFrame(result_dict)
        df.to_csv(f'../fin_data/clickhouse_data/{stock_name}_notice.csv')
        print(stock_name, len(df))

def load_news(config):
    for stock_name in stock_names:
        query_str = f"select publish_time, title, source_url, content from app_news \
            where content like \'%{stock_name}%\' \
            and (length(content) - length(replace(content, \'{stock_name}\', \'\'))) / length(\'{stock_name}\') >= 3 \
            and title is not null \
            and length(content) < 5000"
            # limit 10
        # result_list = client.execute(query_str)
        client = clickhouse_connect.get_client(host=config['host'], username=config['user'], password=config['password'], database=config['database'])
        result_list = client.query(query_str).result_set
        result_dict = {
            'post_publish_time': [],
            'title': [],
            'source_url': [],
            'content': []
        }
        for res in result_list:
            content = res[3]
            if stock_name not in content[:2000]:  
                continue
            result_dict['post_publish_time'].append(res[0])
            result_dict['title'].append(res[1])
            result_dict['source_url'].append(res[2])
            result_dict['content'].append(res[3])
        result_dict['code_name'] = [name2code[stock_name] for i in range(len(result_dict['title']))]
        result_dict['text_type'] = ['news' for i in range(len(result_dict['title']))]
        df = pd.DataFrame(result_dict)
        df.to_csv(f'../fin_data/clickhouse_data/{stock_name}_news.csv')
        print(stock_name, len(df))

if __name__ == "__main__":

    config = {
        'host': "10.213.120.149",
        'port': 8123,
        'user': 'ckuser',
        'password': 'Wande@230',
        'database': 'iir'
    }
    load_notice(config)
    load_news(config)




