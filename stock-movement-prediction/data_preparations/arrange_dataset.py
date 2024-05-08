import pandas as pd
import random

code_df = pd.read_csv('hs_restricted.csv')
stock_names = code_df['SECURITY_NAME_ABBR'].tolist()
stock_codes = code_df['SECUCODE'].tolist()
code2name = {stock_codes[i].lower():stock_names[i] for i in range(len(stock_names))}
print(code2name)

df = pd.read_csv('../fin_data/content_with_labels/all_slope.csv')

grouped_df = df.groupby(['publish_date', 'code_name'])['title'].apply(list).reset_index()

def merge_summary(code, selected_titles):
	selected_titles = ['\''+title+'\'' for title in selected_titles]
	summary = "本日" + code2name[code] + "的相关新闻标题如下：" + ', '.join(selected_titles)
	return summary

def random_select(row):
	titles = row['title']
	
	summary_list = []
	if (len(titles)) <= 5:
		selected_titles = titles
		summary_list.append(merge_summary(row['code_name'], selected_titles))
	else:
		num_groups = len(titles) // 5
		random.shuffle(titles)
		summary_list += [merge_summary(row['code_name'], titles[i*5:(i+1)*5]) for i in range(num_groups)]
		if len(titles) % 5 > 0:
			summary_list.append(merge_summary(row['code_name'], titles[num_groups*5:]))
	return summary_list

def get_label(row):
	labels = list(df[(df['publish_date'] == row['publish_date']) & (df['code_name'] == row['code_name'])]['label'])
	# print(labels)
	assert all(element == labels[0] for element in labels)
	return labels[0]

def get_trend(row):
	trends = list(df[(df['publish_date'] == row['publish_date']) & (df['code_name'] == row['code_name'])]['trend_before'])
	assert all(element == trends[0] for element in trends)
	return trends[0]

grouped_df['summary'] = grouped_df.apply(random_select, axis=1)
grouped_df['label'] = grouped_df.apply(get_label, axis=1)
grouped_df['trend_before'] = grouped_df.apply(get_trend, axis=1)
grouped_df = grouped_df.explode('summary').drop('title', axis=1)

grouped_df.to_csv('../fin_data/content_with_labels/all_grouped_titles_slope.csv', index=False)