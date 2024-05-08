import os

checkpoints = os.listdir('evaluate_new_data_slope_result')
checkpoints.sort()
for checkpoint in checkpoints:
    strict_correct = 0
    loose_correct = 0
    total = 0
    result_path = os.path.join('evaluate_new_data_slope_result', checkpoint, 'generated_predictions.txt')
    if not os.path.exists(result_path):
        continue
    for line in open(result_path, 'r', encoding='utf-8'):
        line = eval(line)
        total += 1
        strict_correct += (line['labels'] == line['predict'])
        loose_correct += (line['labels'].replace('适度减持', '卖出').replace('强力', '') == line['predict'].replace('适度减持', '卖出').replace('强力', ''))
    strict_accuracy = (strict_correct + 0.0) / total * 100
    loose_accuracy = (loose_correct + 0.0) / total * 100
    print(checkpoint, round(strict_accuracy, 4), round(loose_accuracy, 4))