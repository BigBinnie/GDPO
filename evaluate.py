import os
from glob import glob
import scipy
from tqdm import tqdm
import regex as re
import json
import argparse
import torch
from preference_datasets import get_batch_iterator
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import scipy
import torch.nn.functional as F
import numpy as np
import rouge
import datasets

option2number = {
        'Very good': 5,
        'Somewhat good': 4, 
        'Better':4,
        'Yes':4,
        'Same':3, 
        'Worse':2,  
        'No':2, 
        'Somewhat bad':2, 
        'Very bad':1,  
        'DK/Refused':0, 
}
number2option = {
    5: ['Very good'],
    4: ['Somewhat good','Better','Yes'],
    3: ['Same'],
    2: ['Worse','No','Somewhat bad'],
    1: ['Very bad'],
    0: ['DK/Refused']
}

# load option_map from the json file
with open("data/option_map.json", "r") as f:
    option_map = json.load(f)

def get_most_similar_sentence(model, sentence, sentences):
    # print(f"Finding the most similar sentence for: {sentence} in {len(sentences)} sentences...")
    max_similarity = -100
    most_similar_sentence = None
    most_similar_index = None
    for i, s in enumerate(sentences):
        similarity = model.get_score(sentence, s, 'cosine')
        if similarity >= max_similarity:
            max_similarity = similarity
            most_similar_sentence = s
            most_similar_index = i
    return most_similar_sentence, most_similar_index

def get_predictions(file_path, raw_data):
    pred_files = glob(f"{file_path}/predict_*.txt")
    predictions = []
    acc = []
    for pred_file in tqdm(pred_files):
        preds_index = []
        tmp_acc = 0
        with open(pred_file, "r") as f:
            preds = f.readlines()
        print(f"Number of predictions: {len(preds)}")
        for pred, raw in zip(preds, raw_data):
            pred = pred.strip()
            # get the first token
            # convert the token to number
            try:
                op = pred[0]
                op = int(op)
            except:
                op = 0
            preds_index.append(op)
            # use regex to get string wthin the <> tags
            try:
                pred = re.search(r"<.*?>", pred).group()
            except:
                pred = ""
            # pred = re.search(r"<.*?>", pred).group()
            pred = pred[1:-1]
            # get the most similar sentence
            # most_similar_sentence, most_similar_index = get_most_similar_sentence(model, pred, raw["options_map"])
            if pred in option_map:
                option_idx = option_map[pred]
            # option_idx = option2number[raw["options_map"][most_similar_index]]
                if option_idx == op:
                    tmp_acc += 1

        # print("predictions: ", preds_index[:10])
        print(len(preds_index))
        if len(preds_index) != len(raw_data):
            print(f"Length not equal: {len(preds_index)}, {len(raw_data)}")
        else:
            predictions.append(preds_index) 
        acc.append(tmp_acc/len(raw_data))
    # save the predictions
    with open(f"{file_path}/sim_predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
    # print(f"Average accuracy: {sum(acc)/len(acc)}")
    return sum(acc)/len(acc)

def get_commet(file_path, raw_data):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    pred_files = glob(f"{file_path}/predict_*.txt")
    with open(pred_files[0], "r") as f:
        predictions = f.readlines()
    commet_list = []
    for pred,sample in zip(predictions, raw_data):
        try:
            op = pred[0]
            op = int(op)
        except:
            op = 0
        if op not in number2option:
            commet_list.append(0)
            # print(f"Option not found: {op}")
            continue
        options = number2option[op]
        
        try:
            resp = pred.split(">")[1].strip()
        except:
            resp = pred[1:]
        option_map = sample["options_map"]
        label = ""
        for option in options:
            if option in option_map:
                option_idx = option_map.index(option)
                label = sample["answers"][option_idx]
                break
        
        if label!= "":
            # calculate the sentence similarity of the response and the label
            embedding_1= model.encode(resp, convert_to_tensor=True)
            embedding_2 = model.encode(label, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding_1, embedding_2).item()
            commet_list.append(similarity)
        else:
            commet_list.append(0)
    
    print("length of commet: ", len(commet_list))
    return sum(commet_list)/len(commet_list)

def get_rouge(file_path, raw_data):
# open one prediction file
    pred_files = glob(f"{file_path}/predict_*.txt")
    with open(pred_files[0], "r") as f:
        predictions = f.readlines()
    rouge_list = []
    for pred,sample in zip(predictions, raw_data):
        try:
            op = pred[0]
            op = int(op)
        except:
            op = 0
        if op not in number2option:
            # print(f"Option not found: {op}")
            continue
        options = number2option[op]
        
        try:
            resp = pred.split(">")[1].strip()
        except:
            resp = pred[1:]
        option_map = sample["options_map"]
        label = ""
        for option in options:
            if option in option_map:
                option_idx = option_map.index(option)
                label = sample["answers"][option_idx]
                break
        
        if label!= "":
            # calculate the rouge score of the response and the label
            r = rouge.Rouge()
            # print(f"Response: {resp}, Label: {label}")

            if resp.replace(".","").strip() == "":
                rouge_list.append(0)
                continue
            scores = r.get_scores(resp, label)
            # get rouge-l score
            rouge_list.append(scores[0]["rouge-l"]["f"])
        else:
            rouge_list.append(0)
            # print(f"Label not found: {label}")
    
    return sum(rouge_list)/len(rouge_list)

def get_js_distance(file_path, raw_data):
    with open(f"{file_path}/sim_predictions.json", "r") as f:
        predictions = json.load(f)
    dis_list = []
    bias_list = []
    for i, item in enumerate(raw_data):
        # get the distribution of the predictions
        predict_distribution = [0] * 6
        for j in range(len(predictions)):
            # print(f"Predictions: {j},{i}")
            if predictions[j][i] > 5 or predictions[j][i] < 0:
                predict_distribution[0] += 1
            else:
                predict_distribution[predictions[j][i]] += 1
        
        # get the percentage of the predictions
        predict_distribution_1 = [p/len(predictions) for p in predict_distribution]
        distance = scipy.spatial.distance.jensenshannon(predict_distribution_1, item["probabilities"])

        # if distance is not nan value, add it to the list
        if distance == distance:
            dis_list.append(distance)
        else:
            print(f"Predict distribution: {predict_distribution}, True distribution: {item['probabilities']}")
        item["probabilities_predict"] = predict_distribution_1

        max_index = item["probabilities"].index(max(item["probabilities"]))
        max_value = max(item["probabilities"])
        predict_value = predict_distribution_1[max_index]
        majority_bias = predict_value / max_value
        bias_list.append(majority_bias)

    # save the raw_data to a json file, use utf-8 encoding
    with open(f"{file_path}/eval_prediction.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)


    return dis_list, bias_list

def uniform_distribution(raw_data):
    dis_list = []
    for item in raw_data:
        distance = scipy.spatial.distance.jensenshannon(item["probabilities"], [1/6]*6)
        dis_list.append(distance)
    return dis_list

def uniform_distribution_2(raw_data):
    dis_list = []
    for item in raw_data:
        new_probs = [0]*6
        for x, option in enumerate(item["options_map"]):
            # get the number of the option
            number = option2number[option]
            # add the number to the probability list
            new_probs[number] += 1/len(item["selections"])
        distance = scipy.spatial.distance.jensenshannon(item["probabilities"], new_probs)
        dis_list.append(distance)
    return dis_list

def reverse_distribution(raw_data):
    dis_list = []
    for item in raw_data:
        probs = item["probabilities"]
        # get the order of the probabilities
        order = sorted(range(len(probs)), key=lambda k: probs[k])
        # reverse the order
        reverse_order = order[::-1]
        # get the reverse distribution
        reverse_probs = [probs[i] for i in reverse_order]
        distance = scipy.spatial.distance.jensenshannon(probs, reverse_probs)
        dis_list.append(distance)
    return dis_list

def noise_distribution(raw_data, noise):
    # add noise to the max, and mius noise to the min
    dis_list = []
    for item in raw_data:
        probs = item["probabilities"].copy()
        # get the max and min probabilities index
        max_prob = max(probs)
        min_prob = min(probs)
        for i, p in enumerate(probs):
            if p == max_prob:
                probs[i] -= noise
                break
        for i, p in enumerate(probs):
            if p == min_prob:
                probs[i] += noise
                break
        # print(f"Probs: {probs}", item["probabilities"])
        distance = scipy.spatial.distance.jensenshannon(probs,item["probabilities"])
        dis_list.append(distance)
    # print(len(dis_list))
    return dis_list

def majority_distribution(raw_data):
    # add noise to the max, and mius noise to the min
    dis_list = []
    for item in raw_data:
        probs = item["probabilities"].copy()
        # get the max and min probabilities index
        max_prob = probs.index(max(probs))
        # set the max prob to 1, others to 0
        for i, p in enumerate(probs):
            if i == max_prob:
                probs[i] = 1
            else:
                probs[i] = 0
        # print(f"Probs: {probs}", item["probabilities"])
        distance = scipy.spatial.distance.jensenshannon(probs,item["probabilities"])
        dis_list.append(distance)
    # print(len(dis_list))
    return dis_list

def get_avg_js_distance(batch, logits, tokenizer):
    # get the last token of the prompt
    option_logits = logits[:, -1, :]

    option_logits_6 = torch.zeros((logits.shape[0], 6)).to(logits.device)
    for i in range(6):
        token_id = tokenizer.convert_tokens_to_ids(f"{str(i)}")
        option_logits_6[:, i] = option_logits[:, token_id]
    
    JS_divergence = []
    for i in range (len(batch["probabilities"])):
        js = scipy.spatial.distance.jensenshannon(F.softmax(option_logits_6[i], dim=-1).detach().cpu().numpy(), batch["probabilities"][i].cpu().numpy())
        JS_divergence.append(js)
    
    Avg_JS_divergence = np.mean(JS_divergence)
    return Avg_JS_divergence

def get_probs(args):
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    model = AutoModelForCausalLM.from_pretrained('gpt2-large')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # load state from policy.pt
    state = torch.load(args.model)
    model.load_state_dict(state['state'])
    model.cuda()
    model.eval()
    js_list_all = []
    
    data_iterator_kwargs = dict(
        names=[args.datasets],
        tokenizer=tokenizer,
        shuffle=False,
        max_length=2048,
        max_prompt_length=2000,
        sft_mode='sft',
    )

    data_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=args.n_examples, batch_size=1)
    print("Calculating the Jensen-Shannon distance of the model probs...")
    for batch in tqdm(data_iterator):
        with torch.no_grad():
            # inference
            inputs = batch['prompt_input_ids'].cuda()
            # get the probs of the next token
            logits = model(inputs, attention_mask=batch['prompt_attention_mask'].cuda(), return_dict=True).logits
            js = get_avg_js_distance(batch, logits, tokenizer)
            js_list_all.append(js)
    
    return np.mean(js_list_all)             

# def get_dataset_file(dataset, split="test"):
#     # get the dataset json file from huggingface datasets
#     dataset = datasets.load_dataset(dataset, split=split)
#     return dataset.data_files[split]
    
def prepare_data_and_evaluate(args):
    file = f"{args.output_dir}/label.txt"
    with open(file, "r") as f:
        labels = f.readlines()
        # use regex to remove the <> tags
        labels = [label.split(">")[1] for label in labels]

    mma_eval = datasets.load_dataset(args.datasets, args.split)
    # # load the mma_eval.json file
    # with open(test_file, "r") as f:
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

    acc = get_predictions(args.output_dir, raw_data)
    js_distance_list, bias_list = get_js_distance(args.output_dir, raw_data)
    rouge = get_rouge(args.output_dir, raw_data)
    commet = get_commet(args.output_dir, raw_data)

    uniform_dis = uniform_distribution(raw_data)
    print(f"Uniform distribution: {sum(uniform_dis)/len(uniform_dis)}")
    noise_distribution_1 = noise_distribution(raw_data, 0.1)
    print(f"Noise distribution: {sum(noise_distribution_1)/len(noise_distribution_1)}")
    noise_distribution_2 = noise_distribution(raw_data, 0.05)
    print(f"Noise distribution: {sum(noise_distribution_2)/len(noise_distribution_2)}")
    reverse_dis = reverse_distribution(raw_data)
    print(f"Reverse distribution: {sum(reverse_dis)/len(reverse_dis)}")
    print(f"Majority distribution: {sum(majority_distribution(raw_data))/len(majority_distribution(raw_data))}")
    
    return acc, np.mean(js_distance_list), np.mean(bias_list), rouge, commet

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='gpt2-large')
    argparser.add_argument('--output_dir', type=str, default='/out/sft/step-279552')
    argparser.add_argument('--datasets', type=str, default='mmoqa')
    argparser.add_argument('--n_examples', type=int, default=1843)
    argparser.add_argument('--split', type=str, default='test')
    
    args = argparser.parse_args()

    # get the JS distance of model probs
    prob_js = get_probs(args)
    print(f"Model {args.model}: probs Jensen-Shannon distance: {prob_js}")
    # get the JS distance of sampling results
    acc, sampling_js, bias, rouge, commet = prepare_data_and_evaluate(args)

    with open(os.path.join(args.output_dir, "results.json"), "w") as fp:
        json.dump(
            {
                "probs Jensen-Shannon distance": prob_js,
                "sampling Jensen-Shannon distance": sampling_js,
                "Majority-Bias": bias,
                "Class-Belief Consistency": acc,
                "Rouge-L": rouge,
                "Commet": commet
                },
                fp
            )

if __name__ == "__main__":
    main()