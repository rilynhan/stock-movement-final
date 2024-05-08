label_list = ['强力买入', '买入', '观望', '适度减持', '卖出']
result_dict = {l:{i:0 for i in label_list} for l in label_list}
for line in open('/gpfs/radev/project/ying_rex/yz946/finnlp-data/chatglm-model/fin_model/evaluate_slope_result/checkpoint-1300/generated_predictions.txt', 'r', encoding='utf-8').readlines():
    line = eval(line.strip())
    result_dict[line['labels']][line['predict']] += 1

print(result_dict)