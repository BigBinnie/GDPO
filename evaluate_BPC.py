import json
from tqdm import tqdm
import openai
from openai import AzureOpenAI
import argparse
import os
import re
from glob import glob
import datasets

os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""
client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-02-01"
)

class ChatBot:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self):
        completion = client.chat.completions.create(model="gpt-4o", messages=self.messages)
        return completion.choices[0].message.content
    
def get_prompt(prompt, options, answer, selection):
     
    prompt = f"For the following QA pair, does the answer express the opinion of the selection: \"{selection}\"?\n Question: {prompt}\n Answer: {answer}\n Response in the following format:\n Yes or No\n<One Sentence  Explanation>"   
    return prompt

def get_response(prompt, options, answer, selection):
    prompt = get_prompt(prompt, options, answer, selection)
    chatbot = ChatBot("You are a helpful assistant.")
    print(prompt)
    try:
        response = chatbot(prompt)  
        result = parse_response(response)
        print(response, result)
        # if result == 0:
        #     new_answer  = chatbot(post_prompt)
        # else:
        #     new_answer = answer
        # print(new_answer)
    except Exception as e:
        print(e)
        return "", 1
    return response, result

def parse_response(response):
    if len(response.split("\n")) >= 2:
        if "Yes" in response.split("\n")[0]:
            return 1
        else:
            return 0
    else:
        return 1
    
def main():
    # add args
    parser = argparse.ArgumentParser('evaluate response',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--src_dir', type=str, required=True,
                        help='file to test')
    parser.add_argument('-b', '--begin', type=int, default=0, help='start index')
    parser.add_argument('-e', '--end', type=int, default=1140, help='end index')
    parser.add_argument('-p', '--path', type=str, default="/ua/byao24/workspace/Culture-MT/data/", help='save path')
    parser.add_argument('-d', '--save', type=str, default="en-zh/Wikipedia-evaluation.json", help='log file')
    parser.add_argument('--datasets', type=str, default='mmoqa')

    
    args = parser.parse_args()
    print(args)

    openai.api_key = args.key
    save_path = f"{args.save}/response_evaluation.json"
    result_path = f"{args.save}/evaluate_results.json"

    label = f"{args.src_dir}/label.txt"
    with open(label, "r") as f:
        labels = f.readlines()
        # use regex to remove the <> tags
        labels = [re.sub(r"<.*?>", "", label) for label in labels]
    
    mma_eval = datasets.load_dataset(args.datasets, args.split)
    # file = get_dataset_file(args.datasets)
    # # load the mma_eval.json file
    # with open(file, "r") as f:
    #     mma_eval = json.load(f)

    raw_data = []
    for index, label in enumerate(labels):
        is_found = False
        for item in mma_eval:
            if item["accept_response"].strip() in label.strip():
                raw_data.append(item)
                is_found = True
                break
        if not is_found:
            print(f"Label not found: {index}, {label}")

    assert len(raw_data) == len(labels)
    
    predict_file = (f"{args.src_dir}/predict_4.txt")
    with open(predict_file, "r") as f:
        predicts = f.readlines()
    print(f"predict_file {predict_file}")
    print(f"len(raw_data) {len(raw_data)}")
    print(f"len(predicts) {len(predicts)}")
    assert len(predicts) == len(raw_data)

    results = []
    for req, raw in tqdm(zip(predicts[args.begin:args.end], raw_data[args.begin:args.end])):
        req = req.strip()
        # use regex to get string wthin the <> tags
        try:
            pred = re.search(r"<.*?>", req).group()
        except:
            pred = ""
        # pred = re.search(r"<.*?>", pred).group()
        pred = pred[1:-1]
        # get the response after the <> tags
        try:
            resp = req.split(">")[1].strip()
        except:
            resp = req
        response, result = get_response(raw["prompt"], raw["options"], resp, pred)
        print(response, result)
        raw["evaluate_response"] = response
        results.append(result)

    
    # get stats of the results
    stats = {}
    stats["total"] = len(results)
    stats["correct"] = results.count(1)
    stats["incorrect"] = results.count(0)
    stats["accuracy"] = stats["correct"]/stats["total"]
    print(stats)
    # save the results, use utf-8 encoding
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()