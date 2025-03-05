import os
import json
from openai import AzureOpenAI
from datasets import load_dataset
from tqdm import tqdm
import ast

# set up your own API key and endpoint
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""
client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-02-01"
)

def parse_response(response):
    lines = response.split("\n")
    output_response = []
    tmp_dict = {}
    for line in lines:
        if "Style" in line:
            if "style" in tmp_dict:
                output_response.append(tmp_dict)
                tmp_dict = {}
            tmp_dict["style"] = line.split(":")[1].strip()
        if "Q:" in line:
            tmp_dict["question"] = line.split(":")[1].strip()
        if "A" in line:
            arr = line.split(":")
            tmp_dict[arr[0].strip()] = arr[1].strip()
    output_response.append(tmp_dict)
    return output_response

def get_data(country):
    dataset = load_dataset("Anthropic/llm_global_opinions", split="train")
    outputs = []
    for item in dataset:
        selections = item["selections"]
        string = selections.replace("defaultdict(<class 'list'>, ", "").replace(")", "")
        dict= ast.literal_eval(string)
        if country in dict:
            outputs.append({
                "question": item["question"],
                "options": item["options"],
                "selections": dict
            })
    print(len(outputs))
    return outputs

def generate_dialogues(dataset):
    output = []
    for index, item in tqdm(enumerate(dataset)):
        question = item["question"]
        options = item["options"]
        option_list = options[1:-1].split("', ")
        option_list = [option.replace("'", "") for option in option_list]
        number_of_options = len(option_list)
        
        response = client.chat.completions.create(
        model="gpt-35-turbo", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Assume all talks happen in the United States, please generate 5 different styles of one-turn small talks which may happen in different scenarios for the following Question-Choices pair, for each style, please: \n 1. Choose one style name and show it in the first line, for example, Style 1: Casual and Friendly \n2.  Paraphrase the question as Q for each style. For example, Q: What's your stance on defeating terrorism through military force? \n3. Generate {number_of_options} answers from A1 to A{number_of_options}, and each of them reflect one option in the Choices array. \n4. For each answer Ai, include the choice you use to generate at the beginning of the answer by []. For example, A1: [Many of the problems facing our country can be solved by working with other countries] Yes, but in some cases, military action may be necessary to protect our national security. \n5. Don't copy the exact words in the answer or use too uncommon words\n6. Have one empty line after each style's answers, but don't have empty lines within the style.\nThe Question-Choices pair is: \nQuestion: {question}\nChoices: {options}\n please generate small talks which have {number_of_options} answers."},
            ]
        )
        response = response.choices[0].message.content
        print(response)
        response = parse_response(response)
        output.append({
            "question": question,
            "options": option_list,
            "response": response,
            "selections": item["selections"]
        })

        if index % 20 == 0:
            with open("outputs.json", "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
    with open("outputs.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    country = "United States"
    dataset = get_data(country)
    generate_dialogues(dataset)