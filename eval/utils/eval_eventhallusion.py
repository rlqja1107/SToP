import json
from tqdm import tqdm
import sys
sys.path.append(".")
from openai import OpenAI
import argparse




misleading = 'Imagine you are a referee tasked with evaluating a model\'s output. \
The model will output a detailed description of a video. \
You will receive both the model\'s output and the ground truth event. \
Your task is to determine whether the model\'s description is consistent with the ground truth event. \
If you find any other descriptions unrelated to the ground truth event, answer "no." Otherwise, answer "yes." \
You need only focus on the consistency of the event and action. Do not judge the description of specific object, environment, atmosphere, and so on. \
Please answer yes or no in the first word of your reply! Then, provide your analysis and reasoning. \
Model output: {}\
Ground-truth event: {}'

entire = 'Imagine you are a referee tasked with evaluating a model\'s output. \
The model will output a detailed description of a video. \
You will receive both the model\'s output and a ground-truth event. \
Your task is to determine whether the event described in the model\'s output is consistent with the ground-truth event. \
If true, answer "yes." If it is not consistent with the ground-truth event, answer "no." \
You need only focus on the consistency of the event and action. Do not judge the description of specific object, environment, atmosphere, and so on. \
Please answer yes or no in the first word of your reply! Then, provide your analysis and reasoning. \
Model output: {}\
Ground-truth event: {}'

interleave = 'Imagine you are a referee tasked with evaluating a model\'s output. \
The model will output a detailed description of a video. You will receive the output of the tested model and a special event. \
You need to determine whether this special event is mentioned in the output of the model. \
If mentioned, you need to answer "yes", otherwise answer "no". \
You need only focus on the consistency of the event and action. Do not judge the description of specific object, environment, atmosphere, and so on. \
Please answer yes or no in the first word of your reply! Then, provide your analysis. \
Output: {}\
Unexpected event: {}'


def extract_pred(video_llm_output):
    video_llm_output = video_llm_output.lower()
    if video_llm_output.startswith("yes"):
        return "Yes."
    elif video_llm_output.startswith("no"):
        return "No."
    else:
        return None

def binary_eval(predictions):
    total_questions = 0
    total_correct = 0
    for split, pred_dict in predictions.items():
        question_cnt = 0
        correct = 0
        
        for video_key, video_info_with_qa in pred_dict.items():
            for qa in video_info_with_qa['qa']:
                question_cnt += 1
                gt_answer = qa['answer']
                pred = extract_pred(qa['prediction'])
                
                if gt_answer == pred:
                    correct += 1
        total_questions += question_cnt
        total_correct += correct    
        print (f"{split}: ques: {question_cnt}, correct: {correct}, acc: {correct / question_cnt}")
    print (f"overall: ques: {total_questions}, correct: {total_correct}, acc: {total_correct / total_questions}")
    

def get_chat_gpt_response(prompt, client):

    completion = client.chat.completions.create(
    model='gpt-5',
    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] }],
    )
    response = completion.choices[0].message.content
    return response
    

def process_description(video_key, video_data, api_key, prompt): 
    response = get_chat_gpt_response(prompt, api_key)
    
    if 'error' in response:
        video_data['judgement'] = ''
        print(f"video processing: {video_key} fail.")
        return video_key
    else:
        judgement = response.get('choices', [{}])[0].get('message', {}).get('content', 'No judgement available')
        video_data['judgement'] = judgement
        print(f"video processing: {video_key} succeed.")
        return None



def gpt_judge_eval(json_file_path, api_key, output_file_path):
    error_key = []
    for split, videos in data.items():
        for video_key, video_data in videos.items():
            desc = video_data.get('desc', '')
            if not desc:
                print(f"No description found for video {video_key}")
                video_data['judgement'] = ''
                continue
                
            if split == 'interleave':
                prompt = interleave.format(desc, video_data['event_info']['unexpected'])
            elif split == 'entire':
                prompt = entire.format(desc, video_data['event_info']['caption'])
            else:
                prompt = misleading.format(desc, video_data['event_info']['caption'])
                 
            return_video_key = process_description(video_key, video_data, api_key, prompt)
            if return_video_key is not None:
                error_key.append(return_video_key)
                
    
def compute_eventhallusion_qa_result(i, correct_dict_qa):
    eval_type = i['eval_type']
    type_ = i['eval_type'].split("_")[0]
    if '_qa' in eval_type:
        gt = i['gt'].strip().strip(".").lower().strip(",")
        pred = i['pred'].split(" ")[0].strip().strip(".").lower().strip(",")
        if pred not in ['yes', 'no']:
            print(pred)
        correct = 0
        if gt == pred:
            correct_dict_qa.append(1)
            correct += 1
        else:
            correct_dict_qa.append(0)

