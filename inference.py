import torch
from preference_datasets import get_batch_iterator
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
import scipy
import torch.nn.functional as F
import numpy as np

def get_js_distance(batch, logits, tokenizer):
    # get the last token of the prompt
    option_logits = logits[:, -1, :]

    # initialize the logits in a tensor with shape (batch_size, 6)
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

def inference(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    # add padding token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # load state from policy.pt
    state = torch.load(args.ckpt)
    model.load_state_dict(state['state'])
    model.cuda()
    model.eval()

    data_iterator_kwargs = dict(
        names=[args.datasets],
        tokenizer=tokenizer,
        shuffle=False,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        sft_mode='sft',
    )

    print('Start inference', args.begin, args.end)
    for j in range(args.begin, args.end):  
        print(f'Inference on dataset {j}')
        output_list = []
        label_list = []
        data_iterator = get_batch_iterator(**data_iterator_kwargs, split=args.split, batch_size=4, n_examples=args.n_examples)
        count = 0
        for batch in tqdm(data_iterator):
            with torch.no_grad():
                # inference
                inputs = batch['prompt_input_ids'].to("cuda")
                # append a new token to the input
                outputs = model.generate(inputs, max_length=args.max_length, num_return_sequences=1, do_sample=True, attention_mask=batch['prompt_attention_mask'].to("cuda"), pad_token_id=tokenizer.eos_token_id, top_k=0, temperature=1)
            
                # decode outputs
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # remove the prompt from each output
                for i, prompt in enumerate(batch['prompt']):
                    # print(f'Prompt: {prompt}', f'Output: {outputs[i]}')
                    outputs[i] = outputs[i].replace(prompt, '')
                output_list.extend(outputs)
                label_list.extend(batch['chosen_response_only'])
                if count % 2 == 0:
                    # write to file
                    with open(f'{args.output_dir}/predict_{j}.txt', 'w', encoding="utf-8") as f:
                        for output in output_list:
                            f.write(output.strip().replace("\n"," ") + '\n')
                    if j == 0:
                        with open(f'{args.output_dir}/label.txt', 'w', encoding="utf-8") as f:
                            for label in label_list:
                                f.write(label.strip() + '\n')
                
        # write to file
        with open(f'{args.output_dir}/predict_{j}.txt', 'w', encoding="utf-8") as f:
            for output in output_list:
                f.write(output.strip().replace("\n"," ") + '\n')
        if j == 0:
            with open(f'{args.output_dir}/label.txt', 'w', encoding="utf-8") as f:
                for label in label_list:
                    f.write(label.strip() + '\n')
        print(f'Finished inference on dataset {j}')

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--begin', type=int, default=0)
    argparser.add_argument('--end', type=int, default=10)
    argparser.add_argument('--model', type=str, default='gpt2-large')
    argparser.add_argument('--ckpt', type=str, default='')
    argparser.add_argument('--output_dir', type=str, default='')
    argparser.add_argument('--datasets', type=str, default='maop')
    argparser.add_argument('--n_examples', type=int, default=1843)
    argparser.add_argument('--max_length', type=int, default=512)
    argparser.add_argument('--max_prompt_length', type=int, default=256)
    argparser.add_argument('--split', type=str, default='test')
    
    args = argparser.parse_args()
    print(args)
    inference(args)

# main function
if __name__ == '__main__':
    main()
        
    