import datasets
import torch
import json
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from huggingface_hub import login
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data

def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data

def get_mmoqa(name: str, split: str, silent: bool = False, cache_dir: str = None, dpo_mode: bool = False) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the multi-community alignment data from json.
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading {name} dataset ({split} split) from json..., dpo_mode: {dpo_mode}')
    with open(f'data/{name}/mma_{split}.json') as f:
        dataset = json.load(f)

    def split_prompt_and_responses(ex):
        # print("ex:", ex)
        prompt = f"<Q>{ex['prompt']}<A>"
        chosen_response = f"{ex['accept_choice_id']}<{ex['accept_choice']}> {ex['accept_response']}"
        if dpo_mode:
            rejected_response = f"{ex['reject_choice_id']}<{ex['reject_choice']}> {ex['reject_response']}"
        else:
            rejected_response = f"{ex['accept_choice_id']}<{ex['accept_choice']}> {ex['reject_response']}"
        probs = ex["probabilities"]
        return prompt, chosen_response, rejected_response, probs

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing MMA', disable=silent):
        prompt, chosen, rejected, probs = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        data[prompt]['probabilities'] = probs

    return data

def get_movie_reviews(dataset_name: str, split: str, silent: bool = False, cache_dir: str = None, dpo_mode: bool = False) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the multi-community alignment data from json.
       For this dataset, the sft_target is just the chosen response.
    """
    # print(f'Loading multi-community alignment dataset ({split} split) from json..., dpo_mode: {dpo_mode}')
    # with open(f'data/movie_review_small/mma_{split}.json') as f:
    #     dataset = json.load(f)
    print(f'Loading multi-community alignment dataset ({split} split) from Huggingface..., dpo_mode: {dpo_mode}')    
    dataset = datasets.load_dataset(f'{dataset_name}', split=split, cache_dir=cache_dir)
    print('done')
    def split_prompt_and_responses(ex):
        # print("ex:", ex)
        prompt = f"<Q>{ex['prompt']}<A>"
        chosen_response = f"{ex['accept_choice_id']}{ex['accept_response']}"
        if dpo_mode:
            rejected_response = f"{ex['reject_choice_id']}{ex['reject_response']}"
        else:
            rejected_response = f"{ex['accept_choice_id']}{ex['reject_response']}"
        probs = ex["probabilities"]
        return prompt, chosen_response, rejected_response, probs

    data = []
    for row in tqdm.tqdm(dataset, desc='Processing MMA', disable=silent):
        prompt, chosen, rejected, probs = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = 0
        data.append({
            "prompt": prompt,
            "pairs": [(n_responses, n_responses + 1)],
            "responses": responses,
            "sft_target": chosen,
            "probabilities": probs
        })
    
    print("len(data):", len(data))
    return data

def get_mmoqa(dataset_name, split: str, silent: bool = False, cache_dir: str = None, dpo_mode: bool = False) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the multi-community alignment data from json.
       For this dataset, the sft_target is just the chosen response.
    """
    if split not in ['train', 'eval', 'test']:
        print(f'Loading multi-community alignment dataset ({split} split) from json..., dpo_mode: {dpo_mode}')
        with open(f'data/opinion_number_prob_analysis/mma_{split}.json') as f:
            dataset = json.load(f)
    else:
        print(f'Loading multi-community alignment dataset ({split} split) from Huggingface..., dpo_mode: {dpo_mode}')    
        dataset = datasets.load_dataset(f'{dataset_name}', split=split, cache_dir=cache_dir)
        print('done')

    def split_prompt_and_responses(ex):
        if "context" in ex:
            prompt = f"{ex['context']}<Q>{ex['prompt']}<A>"
        else:
            prompt = f"<Q>{ex['prompt']}<A>"
        chosen_response = f"{ex['accept_choice_id']}<{ex['accept_choice']}>{ex['accept_response']}"
        if dpo_mode:
            rejected_response = f"{ex['reject_choice_id']}<{ex['reject_choice']}>{ex['reject_response']}"
        else:
            rejected_response = f"{ex['accept_choice_id']}<{ex['accept_choice']}>{ex['reject_response']}"
        probs = ex["probabilities"]
        return prompt, chosen_response, rejected_response, probs

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing MMA', disable=silent):
        prompt, chosen, rejected, probs = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        data[prompt]['probabilities'] = probs

    return data

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, dpo_mode: bool = False):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif 'mmoqa' in name:
        data = get_mmoqa(name, split, silent=silent, cache_dir=cache_dir, dpo_mode=dpo_mode)
    elif 'movie_reviews' in name:
        data = get_movie_reviews(name, split, silent=silent, cache_dir=cache_dir, dpo_mode=dpo_mode)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return data

def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch

def tokenize_batch_element_kto(prompt: str, generation: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int, prefix: str='target') -> Dict:
        """
        Tokenize a single batch element and truncate if prompt + generation is too long. Batch element is turned into Pytorch 
        tensors in self.collate. Create the labels for the generation, which are of length equal to the sum of the length of 
        the prompt and the generation, with -100 for the prompt tokens.

        Args:
        - conversation: list of previous turns, each resembling dict {"role": "assistant", "content": generation}
        - generation: list of current turns, each resembling dict {"role": "assistant", "content": generation}
        - truncation_mode: one of 'keep_start'/'keep_end' (truncate end/beginning of prompt respectively)
        - prefix: the prefix corresponding to the generation (e.g., 'chosen', 'rejected', 'target')

        Returns:
            A dict of the tokenized prompt and the concatenation of the two on all relevant elements (e.g., tokens, 
            attention mask, etc.). The generation elements will have keys starting with '{prefix}_' and the concatenated 
            elements will have keys starting with '{prefix}_combined_'.
        """
        # print("prompt:", prompt)
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        generation_tokens = tokenizer(generation, add_special_tokens=False)

        assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
        assert tokenizer.eos_token_id not in generation_tokens['input_ids'], f"Generation contains EOS token: {generation}"

        generation_tokens['input_ids'].append(tokenizer.eos_token_id)
        generation_tokens['attention_mask'].append(1)

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens['input_ids']) + len(generation_tokens['input_ids']) > max_length:
            if truncation_mode == 'keep_start':
                prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
            elif truncation_mode == 'keep_end':
                prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # if that's still too long, truncate the response
        if len(prompt_tokens['input_ids']) + len(generation_tokens['input_ids']) > max_length:
            chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

        tokenized_prompt_and_generation_tokens = {k: prompt_tokens[k] + generation_tokens[k] for k in generation_tokens}

        # Prepare the batch element
        batch_element = {
            'prompt_input_ids': prompt_tokens['input_ids'],
            'prompt_text': prompt,
            f'{prefix}_text': generation,
            f'{prefix}_combined_text': prompt + generation,
            f'{prefix}_combined_input_ids': tokenized_prompt_and_generation_tokens['input_ids'],
            f'{prefix}_combined_attention_mask': tokenized_prompt_and_generation_tokens['attention_mask']
        }

        # Prepare labels
        labels = tokenized_prompt_and_generation_tokens['input_ids'][:]
        labels[:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
        batch_element[f'{prefix}_labels'] = labels

        token_id = tokenizer.convert_tokens_to_ids(">")
        # find the token in labels, and print the label which has no token_id
        hasToken = False
        for i, label in enumerate(labels):
            if label == token_id:
                hasToken = True
                break
        if not hasToken:
            print("prompt:", prompt)
            print("prompt_tokens:", prompt_tokens['input_ids'])
            print("labels:", labels)
            print("tokenized_prompt_and_generation_tokens:", tokenized_prompt_and_generation_tokens)
        
        return batch_element

def get_flat_data(args, prompts, full_data):
    """
    Return a flat list of examples given a list of prompts that index self.full_data.
    """
    num_unique = len(full_data) * 2

    allowed_desirable = num_unique * args.get('frac_unique_desirable', 1.0)
    allowed_undesirable = num_unique * args.get('frac_unique_undesirable', 1.0)
    seen_desirable = 0
    seen_undesirable = 0

    flat_data = []
    # print("prompts:", prompts)
    # print(full_data.type)

    for prompt in prompts:
        example = full_data[prompt]
        for p in example["pairs"]:
            if seen_desirable < allowed_desirable:
                flat_data.append((prompt, example["responses"][p[0]], 'chosen', example["probabilities"]))
                seen_desirable += 1
                
            if seen_undesirable < allowed_undesirable:
                flat_data.append((prompt, example["responses"][p[1]], 'rejected', example["probabilities"]))
                seen_undesirable += 1

    return flat_data

def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       dpo_mode: bool=False, method: str = "SFT") -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            if method == "kto":
                full_data = get_dataset(name, split, silent=silent, cache_dir=cache_dir, dpo_mode=dpo_mode)
                prompts = list(full_data.keys())
                args = {"frac_unique_desirable": 1.0, "frac_unique_undesirable": 1.0}
                flat_data = get_flat_data(args, prompts, full_data)
            elif "movie" in name:
                print("name:", name, "no combination")
                for data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, dpo_mode=dpo_mode):
                    # print("data:", data)
                    flat_data.append((data["prompt"], data['responses'], data['pairs'], data['sft_target'], truncation_mode, data['probabilities']))
            else:
                for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, dpo_mode=dpo_mode).items():
                    if "majority" in data:
                        flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode, data['probabilities'], data['majority']))
                    else:
                        flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode, data['probabilities']))
    print(f'Loaded {len(flat_data)} examples from {len(names)} datasets')
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        if method == "kto":
            for prompt, generation, status, probs in flat_data:
                batch_element = tokenize_batch_element_kto(
                    prompt,
                    generation,
                    truncation_mode,
                    tokenizer,
                    max_length,
                    max_prompt_length
                )
                # print(batch_element)
                probs = torch.tensor(probs)
                batch_element['status'] = status
                batch_element['conversation'] = prompt
                batch_element['generation'] = generation 
                batch_element['probabilities'] = probs
                batch.append(batch_element)

                if len(batch) == batch_size:
                    indices = list(range(1, len(batch))) + [0]
                    for i in range(len(batch)):
                        batch[i].update(tokenize_batch_element_kto(
                            batch[i]['conversation'],
                            batch[indices[i]]['generation'],
                            truncation_mode,
                            tokenizer,
                            max_length,
                            max_prompt_length,
                            prefix='KL'
                        ))
                    example_idx += 1
                    yield collate_fn(batch)
                    batch = []
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True
                        break
        else:    
            for prompt, responses, pairs, sft_target, truncation_mode, probs in flat_data:
                if done:
                    break
                if sft_mode:
                    batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                    batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                    # convert probs list to tensor
                    probs = torch.tensor(probs)
                    batch_element['probabilities'] = probs
                    # print("batch_element:", batch_element)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'Finished generating {n_examples} examples on {split} split')
                            done = True

                        batch = []
                else:
                    for p in pairs:
                        if done:
                            break
                        batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                        probs = torch.tensor(probs)
                        batch_element['probabilities'] = probs
                        batch.append(batch_element)
                        example_idx += 1
                        if len(batch) == batch_size:
                            yield collate_fn(batch)
                            if n_examples is not None and example_idx >= n_examples:
                                if not silent:
                                    print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                                done = True
                            batch = []
        if done:
            break

        epoch_idx += 1

def get_batch_iterator_analysis(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       dpo_mode: bool=False) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            if "movie" in name:
                print("name:", name, "no combination")
                for data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, dpo_mode=dpo_mode):
                    # print("data:", data)
                    flat_data.append((data["prompt"], data['responses'], data['pairs'], data['sft_target'], truncation_mode, data['probabilities']))
            else:
                for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, dpo_mode=dpo_mode).items():
                    if "majority" in data:
                        flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode, data['probabilities'], data['majority']))
                    else:
                        flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode), data['probabilities'])
    print(f'Loaded {len(flat_data)} examples from {len(names)} datasets')
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode, probs, majority in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                # convert probs list to tensor
                probs = torch.tensor(probs)
                batch_element['probabilities'] = probs
                batch_element['majority'] = majority
                # print("batch_element:", batch_element)
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                    probs = torch.tensor(probs)
                    batch_element['probabilities'] = probs
                    batch_element['majority'] = majority
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1

def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True