import json
import random
import pandas as pd
from tqdm import tqdm

code_df = pd.read_csv('hs_restricted.csv')
stock_names = code_df['SECURITY_NAME_ABBR'].tolist()
stock_codes = code_df['SECUCODE'].tolist()
code2name = {stock_codes[i].lower():stock_names[i] for i in range(len(stock_names))}

label2target = {
    'very positive': '强力买入',
    'positive': '买入',
    'neutral': '观望',
    'negative': '适度减持',
    'very negative': '卖出'
}

file_path = '../fin_data/content_with_labels/all_grouped_titles_slope.csv'
df = pd.read_csv(file_path)
train_start_date = "2013-01-01"
test_start_date = "2023-03-01"
jsonl_result = {'train': [], 'test': []}
max_instruction = 0
for idx, row in tqdm(df.iterrows(), total=len(df)):
    if row['label'] == 'No data':
        continue
    if row['publish_date'] < train_start_date:
        continue
    elif row['publish_date'] < test_start_date:
        data_type = 'train'
    else:
        data_type = 'test'
    
    instruction_head_set = [
        f"假设你是一位金融专家，请根据以下信息判断今日对股票[{code2name[row['code_name']]}]应采取的操作，", 
        f"请结合股价判断以下新闻对股票[{code2name[row['code_name']]}]做出操作推荐，",
        f"请综合分析股票[{code2name[row['code_name']]}]近日股价及今日相关新闻选择对应的股票操作，",
        f"请预测金融专家对于股票[{code2name[row['code_name']]}]今日应采取的操作，",
        f"下列有关股票[{code2name[row['code_name']]}]的股价信息、新闻对应的股民可能操作如何？"
    ]
    instruction = f"{random.choice(instruction_head_set)}答案从{{强力买入/买入/观望/适度减持/卖出}}中选择\n今天的日期为{row['publish_date']}\n之前10日股价变化趋势为{row['trend_before']}\n{row['summary']}"
    jsonl_result[data_type].append({
        'instruction': instruction,
        'target': label2target[row['label']] 
    })
    if len(instruction) > max_instruction:
        max_instruction = len(instruction)

with open('../fin_data/jsonl/title_train_slope.json', 'w', encoding='utf-8') as f:
    for data in jsonl_result['train']:
        f.write(json.dumps(data, ensure_ascii=False)+'\n')
with open('../fin_data/jsonl/title_test_slope.json', 'w', encoding='utf-8') as f:
    for data in jsonl_result['test']:
        f.write(json.dumps(data, ensure_ascii=False)+'\n')
