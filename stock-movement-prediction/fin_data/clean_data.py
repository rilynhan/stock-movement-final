import os
import re
import pandas as pd
from bs4 import BeautifulSoup

pattern = r'<p>(.*?)</p>'
strong_pattern = re.compile(r'<strong>(.*?)</strong>')
link_pattern = re.compile(r'<ahref=(.*?)>(.*?)</a>')

def get_abstract(origin_html):
    html = origin_html
    soup = BeautifulSoup(html, 'html.parser')
    data_abstract_div = soup.find('div', {'data-type': 'abstract'})
    if data_abstract_div: 
        data_abstract = data_abstract_div.get('data-abstract', '')
        if len(data_abstract) > 5:
            return data_abstract.replace('<br>', '')
    html = html.replace(' ', '').replace('　', '').replace('<br>', '')
    html = html.replace('<p></p>', '')
    p_tags_contents = re.findall(pattern, html)
    if not p_tags_contents:
        return '-'
    content = " ".join(p_tags_contents)
    content = strong_pattern.sub('', content)
    content = link_pattern.sub('', content)
    if len(content) < 5:
        print(origin_html)
        return('-')
    return content

folder_path = "./scrape/results_with_content"
cleaned_folder_path = "./results_with_content_clean"
for file_name in os.listdir(folder_path):
    df = pd.read_csv(os.path.join(folder_path, file_name))
    df["content"] = df.apply(lambda x:get_abstract(x['post_content']), axis=1)
    origin_len = len(df)
    df = df[df['post_publish_time'] != '-']
    df = df[df['content'] != '-']
    df = df[~df['content'].str.contains('')]
    cleaned_len = len(df)
    columns_needed = ['post_publish_time', 'code_name', 'text_type', 'title', 'content']
    for c in df.columns:
        if c not in columns_needed:
            df.drop(c, axis=1, inplace=True)
    df.to_csv(os.path.join(cleaned_folder_path, file_name))
    print(file_name, origin_len, cleaned_len, origin_len-cleaned_len)

'''
1256 1256 0
3053 3045 8
3232 3222 10
7366 7355 11
7877 7866 11
'''