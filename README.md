
# CSPC 452 Final Project: Enhancing Financial Forecasting Models with ChatGLM on Scraped Data

This repository contains all code utilized for our final project. All data scraping, data processing, ChatGLM p-tuning, and evaluations can be reproduced using the following steps.

## Data Scraping and Preparation
The corresponding code is located in the `data_preparations` directory.
- **Environment dependencies**: You can install them directly by running `pip install -r requirements.txt`.

### Update and Download Data
- Update `hs_restricted.csv` with the stocks needed.
- Run `download_titles.py` to scrape news data. The data will be stored in `fin_data/scrape/results`. Then use `download_contents.py` to download all content and `fin_data/clean_data.py` to clean it. The cleaned data will be stored in `fin_data/results_with_content_clean`.

### Data Processing
- Run `read_clickhouse_data.py` to read relevant news and announcements from Clickhouse; the data will be stored in `fin_data/clickhouse_data`.
- Run `add_labels.py` to add labels and stock price information to each piece of news/announcement. The data is stored in `fin_data/content_with_labels`.
  - You can adjust `THRESH_VERY`, `THRESH`, `FORWARD_DAYS`, `END_DATE`.
  - The code uses akshare to read stock information from January 1, 2012, to `END_DATE`.
  - Use `UnivariateSpline` to fit the stock price curve. If you want to observe the fitted curve, you can run `fin_data/candle_figure.py` to view it.
  - Labels are defined as the slope of the stock price curve on the nearest stock trading day after the news/announcement is released:
    - `s < -THRESH_VERY`: very_negative (sell)
    - `-THRESH_VERY <= s < -THRESH`: negative (moderate sell)
    - `-THRESH <= s <= THRESH`: neutral (hold)
    - `THRESH < s <= THRESH_VERY`: positive (buy)
    - `s > THRESH_VERY`: very_positive (strong buy)
  - Add the slope of the stock price curve `FORWARD_DAYS` days before the news/announcement as judgment information.

### Dataset Arrangement
- Run `arrange_dataset.py` to bundle news information from the same day. The data is stored in `fin_data/content_with_labels`.
  - The current packaging method is: if a stock has more than 5 news + announcement items on the same day, they are randomly packaged into multiple groups of 5 each.

### Dataset Conversion
- Run `making_dataset.py` to convert the dataset into jsonl format. The dataset is stored in `fin_data/jsonl`.
  - Set the data between `train_start_date` and `test_start_date` as the training set, and data after `test_start_date` as the test set.


## ChatGLM2-6B p-tuning
1. Clone the code for [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) locally and install the corresponding dependencies.
2. You can choose to download the parameters for ChatGLM2 from [huggingface](https://huggingface.co/THUDM/chatglm2-6b), which might be faster than having the code automatically pull them from huggingface during runtime.
3. Enter the `ptuning` folder of the ChatGLM2-6B code, and run `train.sh`. Before running, you need to modify the following parameters:
    * `train_file` and `validation_file` should point to the previously processed jsonl files.
    * Set `prompt_column` to `instruction`.
    * Set `response_column` to `target`.
    * `model_name_or_path` should be set to the location of the downloaded ChatGLM2 parameters.
    * `output_dir` should be set to the location where the p-tuning parameters are to be saved.
    * `max_source_length` should be set as it was previously in `making_dataset.py`.
    * Set `quantization_bit` to 8.
4. After training, you can test using `evaluate.sh`, which we have modified as follows for reference:
    ```python

    PRE_SEQ_LEN=128
    NUM_GPUS=1
   
    for ((STEP = 100; STEP <= 3000; STEP += 100));
    do
        torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
            --do_predict \
            --validation_file /gpfs/radev/project/ying_rex/yz946/stock-movement-final/fin_data/jsonl/title_test_slope.json \
            --test_file /gpfs/radev/project/ying_rex/yz946/stock-movement-final/fin_data/jsonl/title_test_slope.json \
            --overwrite_cache \
            --prompt_column instruction \
            --response_column target \
            --model_name_or_path /gpfs/radev/project/ying_rex/yz946/stock-movement-final/chatglm-model/chatglm2-6b \
            --ptuning_checkpoint /gpfs/radev/project/ying_rex/yz946/stock-movement-final/chatglm-model/chatglm2-6b/slope_chatglm2-6b-pt-128-2e-2/checkpoint-$STEP \
            --output_dir /gpfs/radev/project/ying_rex/yz946/stock-movement-final/chatglm-model/fin_model/evaluate_slope_result/checkpoint-$STEP \
            --overwrite_output_dir \
            --max_source_length 460 \
            --max_target_length 8 \
            --per_device_eval_batch_size 1 \
            --predict_with_generate \
            --pre_seq_len $PRE_SEQ_LEN \
            --quantization_bit 8
    done
    ```
5. You can calculate the accuracy by running `evaluation/calculate_accuracy.py` and the actual label distribution under each label by running `evaluation/calculate_label_num.py`. Remember to modify the corresponding file paths before running; you can also generate directly on the web page using `web_demo.py` in the ChatGLM2-6b/ptuning folder.
