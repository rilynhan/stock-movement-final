from openai import OpenAI
import os
import json

def prepare_jsonl(data_file, output_file):
    with open(data_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()  

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            if line.strip():  
                entry = json.loads(line)  
                prompt = entry['instruction']
                completion = entry['target'] + "\n"
                json_record = {"prompt": prompt, "completion": completion}
                json.dump(json_record, outfile) 
                outfile.write('\n')

def upload_file(client, file_name, purpose):
    with open(file_name, "rb") as file:
        response = client.files.create(
            file=file,
            purpose=purpose
        )
    return response.id

def create_fine_tune_job(client, training_file_id, validation_file_id):
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="davinci-002", 
        hyperparameters={
            "n_epochs": 4,
            "batch_size": 4,
            "learning_rate_multiplier": 0.1
        }
    )
    return response.id

def check_job_status(client, job_id):
    status = client.fine_tuning.jobs.retrieve(job_id).status
    print(f'Job status: {status}')
    return status

def main():
    client = OpenAI(
    api_key='sk-proj-d45zoV0dSto8p1PGGX1AT3BlbkFJn3SVbr7NZPTLD7Caa1X5',
    )

    prepare_jsonl('/gpfs/radev/project/ying_rex/yz946/stock-movement-prediction/fin_data/jsonl/title_train_slope.json', 'training_data.jsonl')
    prepare_jsonl('/gpfs/radev/project/ying_rex/yz946/stock-movement-prediction/fin_data/jsonl/title_test_slope.json', 'validation_data.jsonl')

    training_file_id = upload_file(client, 'training_data.jsonl', 'fine-tune')
    validation_file_id = upload_file(client, 'validation_data.jsonl', 'fine-tune')

    job_id = create_fine_tune_job(client, training_file_id, validation_file_id)
    print(f"Fine-tuning job started with ID: {job_id}")


    status = check_job_status(client, job_id)

if __name__ == '__main__':
    main()
