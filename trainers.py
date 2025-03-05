import scipy.spatial
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
import scipy
from omegaconf import DictConfig
import numpy as np
import shutil

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

def preference_loss_majority(batch: Dict[str, Union[List, torch.LongTensor]],  policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    # only get the chosen rewards and rejected rewards of the majority
    chosen_rewards_majority = []
    rejected_rewards_majority = []
    for i in range(len(batch["majority"])):
        if batch["majority"][i] == 1:
            chosen_rewards_majority.append(chosen_rewards[i])
            rejected_rewards_majority.append(rejected_rewards[i])
    if len(chosen_rewards_majority) == 0:
        chosen_rewards_majority = torch.tensor([0.0]).to(chosen_rewards.device)
        rejected_rewards_majority = torch.tensor([0.0]).to(rejected_rewards.device)
    else:
        chosen_rewards_majority = torch.stack(chosen_rewards_majority)
        rejected_rewards_majority = torch.stack(rejected_rewards_majority)
    # chosen_rewards_majority = torch.stack(chosen_rewards_majority)
    # rejected_rewards_majority = torch.stack(rejected_rewards_majority)

    # get the rewards of the minority
    chosen_rewards_minority = []
    rejected_rewards_minority = []
    for i in range(len(batch["majority"])):
        if batch["majority"][i] == 0:
            chosen_rewards_minority.append(chosen_rewards[i])
            rejected_rewards_minority.append(rejected_rewards[i])
    if len(chosen_rewards_minority) == 0:
        chosen_rewards_minority = torch.tensor([0.0]).to(chosen_rewards.device)
        rejected_rewards_minority = torch.tensor([0.0]).to(rejected_rewards.device)
    else:
        chosen_rewards_minority = torch.stack(chosen_rewards_minority)
        rejected_rewards_minority = torch.stack(rejected_rewards_minority)
    print("chosen_rewards_majority", chosen_rewards_majority)
    print("chosen_rewards_minority", rejected_rewards_minority)
    return losses, chosen_rewards, rejected_rewards, chosen_rewards_majority, rejected_rewards_majority, chosen_rewards_minority, rejected_rewards_minority

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def _get_batch_logps_belief(batch: Dict[str, Union[List, torch.LongTensor]],logits: torch.FloatTensor, labels: torch.LongTensor, tokenizer: transformers.PreTrainedTokenizer, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids'
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    token_id = tokenizer.convert_tokens_to_ids(">")
    end_position = []
    for i, sequence in enumerate(labels):
        # Find the last occurrence of ">"
        last_position = (sequence == token_id).nonzero(as_tuple=True)[0]
        # change tensor to int
        if last_position.shape[0] == 0:
            mask_position = torch.argmax((sequence != -100).double(), dim=0)
            print("mask_position", mask_position)
            print("labels", labels[i])
            last_position = torch.tensor([mask_position])
        last_position = last_position.max()
        last_position = last_position.item()
        end_position.append(last_position)


    # set labels to 0 after the second ">" token
    for i in range(labels.shape[0]):
        new_loss_mask = loss_mask[i].clone()
        new_loss_mask[end_position[i]:] = False
        if new_loss_mask.sum().item() > 0:
            loss_mask[i, end_position[i]:] = False

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def kto_get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor):
        """Compute the token-level log probabilities of the given labels under the given logits."""
        # ignoring vocab size, batch size x length should be equal
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        distribution_logps = logits.float().log_softmax(-1)
        per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return per_token_logps * loss_mask
    
def _CE_loss_test(batch: Dict[str, Union[List, torch.LongTensor]], logits: torch.FloatTensor, tokenizer: transformers.PreTrainedTokenizer) -> torch.FloatTensor:
    """Compute the cross-entropy loss for the given batch of inputs.
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids', 'prompt_input_ids','probabilities'.
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
    """
    prompt_length = batch["prompt_input_ids"].shape[1]
    # print("prompt_length", prompt_length)
    chosen_start = prompt_length
    mask_position = torch.argmax((batch["chosen_labels"] != -100).double(), dim=1)
    # minus 1 to get the last token of the prompt
    mask_position = mask_position - 1
    chosen_start = mask_position
    # get the logits of the chosen start token, shape of chosen_start is (batch_size), the output shape should be (batch_size, 1, vocab_size)
    # print("out", batch["chosen_labels"].shape, logits.shape)
    option_logits = logits[torch.arange(logits.shape[0]), chosen_start].unsqueeze(1)

    # initialize the logits in a tensor with shape (batch_size, len(batch["probabilities"]))
    option_logits_6 = torch.zeros((logits.shape[0], len(batch["probabilities"][0]))).to(logits.device)
    for i in range(len(batch["probabilities"][0])):
        token_id = tokenizer.convert_tokens_to_ids(f"{str(i)}")
        option_logits_6[:, i] = option_logits[:, 0, token_id]
    
    # print("option_logits_6", F.log_softmax(option_logits_6))
    # print("batch[probabilities]", batch["probabilities"][:10])
    batch["probabilities"] = torch.stack(batch["probabilities"]).to(logits.device)
    # CE_losses = F.cross_entropy(option_logits_6, batch["probabilities"])
    CE_losses = F.kl_div(F.log_softmax(option_logits_6, dim=1), batch["probabilities"], reduction='batchmean')
    
    JS_divergence = []
    for i in range (batch["probabilities"].shape[0]):
        js = scipy.spatial.distance.jensenshannon(F.softmax(option_logits_6[i], dim=-1).detach().cpu().numpy(), batch["probabilities"][i].cpu().numpy())
        JS_divergence.append(js)
    Avg_JS_divergence = np.mean(JS_divergence)
    # print("Avg_JS_divergence", JS_divergence)
    # change the JS_divergence to tensor
    Avg_JS_divergence = torch.tensor(Avg_JS_divergence).to(CE_losses.device)
    # print("Avg_JS_divergence", Avg_JS_divergence)
    # print("CE_losses", CE_losses)
    return CE_losses, Avg_JS_divergence

def _CE_loss(batch: Dict[str, Union[List, torch.LongTensor]], logits: torch.FloatTensor, tokenizer: transformers.PreTrainedTokenizer) -> torch.FloatTensor:
    """Compute the cross-entropy loss for the given batch of inputs.
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids', 'prompt_input_ids','probabilities'.
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
    """
    prompt_length = batch["prompt_input_ids"].shape[1]
    chosen_start = prompt_length
    if "chosen_labels" in batch:
        mask_position = torch.argmax((batch["chosen_labels"] != -100).double(), dim=1)
        # minus 1 to get the last token of the prompt
    else:
        mask_position = torch.argmax((batch["target_labels"] != -100).double(), dim=1)

    # minus 1 to get the last token of the prompt
    mask_position = mask_position - 1
    chosen_start = mask_position
    option_logits = logits[torch.arange(logits.shape[0]), chosen_start].unsqueeze(1)

    softmax_option_logits = F.softmax(option_logits, dim=-1)
    log_softmax_option_logits = F.log_softmax(option_logits, dim=-1)

    option_logits_6 = torch.zeros((logits.shape[0], len(batch["probabilities"][0]))).to(logits.device)
    softmax_option_logits_6 = torch.zeros((logits.shape[0], len(batch["probabilities"][0]))).to(logits.device)
    for i in range(len(batch["probabilities"][0])):
        token_id = tokenizer.convert_tokens_to_ids(f"{str(i)}")
        option_logits_6[:, i] = log_softmax_option_logits[:, 0, token_id]
        softmax_option_logits_6[:, i] = softmax_option_logits[:, 0, token_id]
    
    # get the sum of each row of option_logits_6
    sum_option_logits_6 = torch.sum(softmax_option_logits_6, dim=1)
    
    batch["probabilities"] = torch.stack(batch["probabilities"]).to(logits.device)

    # get the log_softmax of option_logits_6
    CE_losses = F.kl_div(option_logits_6, batch["probabilities"], reduction='batchmean')
    
    JS_divergence = []
    for i in range (batch["probabilities"].shape[0]):
        js = scipy.spatial.distance.jensenshannon(softmax_option_logits_6[i].detach().cpu().numpy(), batch["probabilities"][i].cpu().numpy())
        JS_divergence.append(js)
    Avg_JS_divergence = np.mean(JS_divergence)

    # change the JS_divergence to tensor
    Avg_JS_divergence = torch.tensor(Avg_JS_divergence).to(CE_losses.device)

    return CE_losses, Avg_JS_divergence, sum_option_logits_6

def _CE_loss_all(batch: Dict[str, Union[List, torch.LongTensor]], logits: torch.FloatTensor, tokenizer: transformers.PreTrainedTokenizer) -> torch.FloatTensor:
    """Compute the cross-entropy loss for the given batch of inputs.
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids', 'prompt_input_ids','probabilities'.
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
    """
    prompt_length = batch["prompt_input_ids"].shape[1]
    # get the location of the first token which is not -100
    mask_position = torch.argmax((batch["chosen_labels"] != -100).double(), dim=1)
    chosen_start = mask_position
    # get labels of the chosen start token  
    # get the logits of the chosen start token, shape of chosen_start is (batch_size), the output shape should be (batch_size, 1, vocab_size)
    option_logits = logits[torch.arange(logits.shape[0]), chosen_start].unsqueeze(1)
    
    # get the logits of the first token of the prompt
    option_logits = option_logits[:, 0, :]
    
    batch["probabilities"] = torch.stack(batch["probabilities"]).to(logits.device)
    # initialize the logits in a tensor with shape (batch_size, vocab_size)
    prob_label = torch.zeros((logits.shape[0], option_logits.shape[1])).to(logits.device)
    for i in range(6):
        token_id = tokenizer.convert_tokens_to_ids(f"<{str(i)}>")
        prob_label[:, token_id] = batch["probabilities"][:, i]

    # CE_losses = F.cross_entropy(option_logits, prob_label)
    KL_divergence = F.kl_div(F.log_softmax(option_logits, dim=1), prob_label, reduction='batchmean')
    CE_losses = KL_divergence

    JS_divergence = []
    for i in range (batch["probabilities"].shape[0]):
        js = scipy.spatial.distance.jensenshannon(F.softmax(option_logits[i], dim=0).detach().cpu().numpy(), prob_label[i].cpu().numpy())
        JS_divergence.append(js)
    Avg_JS_divergence = np.mean(JS_divergence)

    # change the JS_divergence to tensor
    Avg_JS_divergence = torch.tensor(Avg_JS_divergence).to(CE_losses.device)
    return CE_losses, Avg_JS_divergence    

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

def concatenated_inputs_exclude_number(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            # add one more mask token to the end of the chosen_labels
            if 'labels' in k:
                # get the length of prompt
                # prompt_length = batch["prompt_input_ids"].shape[1]
                mask_position = torch.argmax((batch[k] == -100).double(), dim=1)
                # mask the next token of the last token of the prompt
                # mask_position = prompt_length
                # print("mask_position", mask_position)
                # change the next token to -100
                batch[k][torch.arange(batch[k].shape[0]), mask_position] = -100

            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            if 'labels' in k:
                # get the position of the last -100 token
                mask_position = torch.argmax((batch[k] == -100).double(), dim=1)
                # change the next token to -100
                batch[k][torch.arange(batch[k].shape[0]), mask_position] = -100
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

def concatenated_inputs_exclude_belief(batch: Dict[str, Union[List, torch.LongTensor]], tokenizer: transformers.PreTrainedTokenizer) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            # add one more mask token to the end of the chosen_labels
            if 'labels' in k:
                token_id = tokenizer.convert_tokens_to_ids(">")
                end_position = []
                for sequence in batch["chosen_input_ids"]:
                    last_position = (sequence == token_id).nonzero(as_tuple=True)[0].max()
                    last_position = last_position.item()
                    end_position.append(last_position)
                for i in range(batch[k].shape[0]):
                    tmp = batch[k][i].clone()
                    tmp[:end_position[i]] = -100
                    # if all the tokens of tmp are -100, then just keep the old value
                    if torch.sum(tmp == -100) != tmp.shape[0]:
                        batch[k][i] = tmp
                
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            if 'labels' in k:
                token_id = tokenizer.convert_tokens_to_ids(">")
                end_position = []
                for sequence in batch["chosen_input_ids"]:
                    last_position = (sequence == token_id).nonzero(as_tuple=True)[0].max()
                    last_position = last_position.item()
                    end_position.append(last_position)
                for i in range(batch[k].shape[0]):
                    tmp = batch[k][i].clone()
                    tmp[:end_position[i]] = -100
                    # if all the tokens of tmp are -100, then just keep the old value
                    if torch.sum(tmp == -100) != tmp.shape[0]:
                        batch[k][i] = tmp
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

import torch.nn as nn
def resize_token_embeddings(model: nn.Module, new_num_tokens: int):
    """Resize the token embeddings of the given model to the given number of tokens."""
    old_num_tokens = model.get_input_embeddings().num_embeddings
    new_embedding = nn.Embedding(new_num_tokens, model.get_input_embeddings().embedding_dim)
    new_embedding.weight.data[:old_num_tokens] = model.get_input_embeddings().weight.data
    # fill the new embeddings with the mean of the old embeddings
    new_embedding.weight.data[old_num_tokens:] = model.get_input_embeddings().weight.mean(dim=0)
    new_embedding.to(model.get_input_embeddings().weight.device)
    model.set_input_embeddings(new_embedding)

    # resize the output layer
    if hasattr(model, 'lm_head'):
        old_lm_head = model.lm_head
        # get the number of old output vocab size
        old_num_tokens = old_lm_head.out_features
        new_lm_head = nn.Linear(old_lm_head.in_features, new_num_tokens, bias=False)
        #copy the old weights to the new weights
        new_lm_head.weight.data[:old_num_tokens] = old_lm_head.weight.data
        
        # fill the new embeddings with the mean of the old embeddings
        new_lm_head.weight.data[old_num_tokens:] = old_lm_head.weight.mean(dim=0)
        new_lm_head.to(old_lm_head.weight.device)
        model.lm_head = new_lm_head
    else:
        print("No lm_head in the model")

def exclude_belief_from_batch(batch: Dict[str, Union[List, torch.LongTensor]], tokenizer) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            # add one more mask token to the end of the chosen_labels
            if 'labels' in k:
                token_id = tokenizer.convert_tokens_to_ids(">")
                end_position = []
                for i, sequence in enumerate(batch[k]):
                    last_position = (sequence == token_id).nonzero(as_tuple=True)[0]
                    # if last_position.shape[0] == 0:
                    #     mask_position = torch.argmax((sequence != -100).double(), dim=0)
                    #     print(k)
                    #     print("labels", sequence)
                    #     print("target_labels", batch['target_combined_input_ids'][i])
                    #     print("prompt_input_ids", batch['prompt_input_ids'][i])
                    #     last_position = torch.tensor([mask_position])
                    # print("last_position", last_position)
                    last_position = last_position.max()
                    last_position = last_position.item()
                    end_position.append(last_position)
                
                for i in range(batch[k].shape[0]):
                    tmp = batch[k][i].clone()
                    tmp[:end_position[i]+1] = -100
                    # if all the tokens of tmp are -100, then just keep the old value
                    if torch.sum(tmp == -100) != tmp.shape[0]:
                        batch[k][i] = tmp   
    return batch

class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model
        self.policy_dtype = next(policy.parameters()).dtype

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, seed=seed, dpo_mode=(config.loss.name=='dpo'), method=config.loss.name)
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='eval', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, seed=seed, dpo_mode=(config.loss.name=='dpo'), method=config.loss.name)
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')
        # self.eval_iterator_0 = get_batch_iterator(**data_iterator_kwargs, split='eval_0', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, seed=seed, dpo_mode=(config.loss.name=='dpo'))
        # self.eval_batches_0 = list(self.eval_iterator_0)
        # rank0_print(f'Loaded eval_0 data iterator')
        # self.eval_iterator_1 = get_batch_iterator(**data_iterator_kwargs, split='eval_1', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, seed=seed, dpo_mode=(config.loss.name=='dpo'))
        # self.eval_batches_1 = list(self.eval_iterator_1)
        # rank0_print(f'Loaded eval_1 data iterator')
        # self.eval_iterator_2 = get_batch_iterator(**data_iterator_kwargs, split='eval_2', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, seed=seed, dpo_mode=(config.loss.name=='dpo'))
        # self.eval_batches_2 = list(self.eval_iterator_2)


    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo', 'mpo', 'cdpo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo', 'mpo', 'cdpo', 'kto'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], exclude_belief=False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        if not exclude_belief:
            concatenated_batch = concatenated_inputs(batch)
        else:
            concatenated_batch = concatenated_inputs_exclude_belief(batch, self.tokenizer)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def kto_loss(self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:  # Fixed return type
        """Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities."""

        KL_rewards = policy_KL_logps.sum(-1) - reference_KL_logps.sum(-1)
        KL = all_gather_if_needed(KL_rewards.detach(), self.rank, self.world_size).mean().clamp(min=0)
        # Desirable case
        if policy_chosen_logps.shape[0] != 0:
            chosen_rewards = (policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1))
            chosen_losses = self.config.loss.desirable_weight * (1 - F.sigmoid(self.config.loss.beta * (chosen_rewards - KL)))
        else:
            chosen_losses = torch.empty(0, device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)
            chosen_rewards = torch.empty(0, device=policy_chosen_logps.device, dtype=policy_chosen_logps.dtype)

        # Undesirable case
        if policy_rejected_logps.shape[0] != 0:
            rejected_rewards = (policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1))
            rejected_losses = self.config.loss.undesirable_weight * (1 - F.sigmoid(self.config.loss.beta * (KL - rejected_rewards)))
        else:
            rejected_losses = torch.empty(0, device=policy_rejected_logps.device, dtype=policy_rejected_logps.dtype)
            rejected_rewards = torch.empty(0, device=policy_rejected_logps.device, dtype=policy_rejected_logps.dtype)

        losses = torch.cat((chosen_losses, rejected_losses), dim=0)
        
        return losses, chosen_rewards.detach(), rejected_rewards.detach(), KL.detach()

    def kto_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], use_cache: bool=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs.
        
        Args:
            - model: the model to use for the forward pass
            - batch: the microbatch (should have the input ids, attention mask, and labels)
            - use_cache: if true, can get cached logprobs instead

        Returns:
            chosen_logps: log probabilities of chosen examples
            rejected_logps: log probabilities of rejected examples
            KL_logps: log probabilities of the unmatched y'|x (used to estimate the KL divergence between policy and reference)
        """
        # with self.accelerator.autocast():
        with torch.no_grad():
            if use_cache:
                KL_logps = model(batch[f'KL_combined_input_ids']).to(self.policy_dtype)
            else:
                KL_logits = model(
                    batch[f'KL_combined_input_ids'],
                    attention_mask=batch[f'KL_combined_attention_mask']
                ).logits.to(self.policy_dtype)

                KL_logps = kto_get_batch_logps(KL_logits, batch[f'KL_labels'])

        if use_cache:
            target_logps = model(batch[f'target_combined_input_ids']).to(self.policy_dtype)
        else:
            target_logits = model(
                batch[f'target_combined_input_ids'],
                attention_mask=batch[f'target_combined_attention_mask']
            ).logits.to(self.policy_dtype)

            target_logps = kto_get_batch_logps(target_logits, batch[f'target_labels'])

        assert target_logps.shape[0] == len(batch['status'])
        chosen_idx = [i for i in range(target_logps.shape[0]) if batch['status'][i] == 'chosen']
        rejected_idx = [i for i in range(target_logps.shape[0]) if batch['status'][i] == 'rejected']
        chosen_logps = target_logps[chosen_idx, ...]
        rejected_logps = target_logps[rejected_idx, ...]

        return chosen_logps, rejected_logps, KL_logps
    
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True, eval_name: str = 'train'):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else eval_name

        if loss_config.name in {'dpo', 'ipo', 'cdpo'}:
            with torch.no_grad():
                policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
                CE_losses, JS_divergence, cls_sum = _CE_loss(batch, policy_chosen_logits, self.tokenizer)
            
            if loss_config.name == 'cdpo':
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch, exclude_belief=True)
            else:
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            
            with torch.no_grad():
                if loss_config.name == 'cdpo':
                    reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch, exclude_belief=True)
                else:
                    print("reference model device", next(self.reference_model.parameters()).device)
                    reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            if loss_config.name == 'dpo' or loss_config.name == 'cdpo':
                loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
            elif loss_config.name == 'ipo':
                loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
            else:
                raise ValueError(f'unknown loss {loss_config.name}')

            losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            # losses, chosen_rewards, rejected_rewards, chosen_rewards_majority, rejected_rewards_majority, chosen_rewards_minority, rejected_rewards_minority = preference_loss_majority(batch, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)


            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=True)
            CE_losses, JS_divergence, cls_sum = _CE_loss(batch, policy_chosen_logits, self.tokenizer)
            losses = -policy_chosen_logps
        
        elif loss_config.name == 'mft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps_belief(batch, policy_chosen_logits, batch['chosen_labels'], tokenizer= self.tokenizer, average_log_prob=False)
            
            CE_losses, JS_divergence, cls_sum = _CE_loss(batch, policy_chosen_logits, self.tokenizer)
            policy_logps_belief = _get_batch_logps_belief(batch, policy_chosen_logits, batch['chosen_labels'], self.tokenizer, average_log_prob=True)
            policy_logps_belief = all_gather_if_needed(policy_logps_belief.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/chosen-belief'] = policy_logps_belief.cpu().numpy().tolist()
            # losses = -policy_chosen_logps + CE_losses
            losses = CE_losses - policy_logps_belief
        
        elif loss_config.name == 'mpo':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            CE_losses, JS_divergence, cls_sum = _CE_loss(batch, policy_chosen_logits, self.tokenizer)
            policy_logps_belief = _get_batch_logps_belief(batch, policy_chosen_logits, batch['chosen_labels'], self.tokenizer, average_log_prob=True)

            # calculate the dpo loss
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch, exclude_belief=True)
            
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch, exclude_belief=True)
            loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}

            dpo_losses, chosen_rewards, rejected_rewards = preference_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
            
            losses = CE_losses + dpo_losses - policy_logps_belief
            
            policy_logps_belief = all_gather_if_needed(policy_logps_belief.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/chosen-belief'] = policy_logps_belief.cpu().numpy().tolist()
        
        elif loss_config.name == 'kto':
            policy_chosen_logits = self.policy(
                batch['target_combined_input_ids'], 
                attention_mask=batch['target_combined_attention_mask'],
            ).logits.to(self.policy_dtype)

            CE_losses, JS_divergence, cls_sum = _CE_loss(batch, policy_chosen_logits, self.tokenizer)
            policy_logps_belief = _get_batch_logps_belief(batch, policy_chosen_logits, batch['target_labels'], self.tokenizer, average_log_prob=True)
            batch = exclude_belief_from_batch(batch, self.tokenizer)
            policy_chosen_logps, policy_rejected_logps, policy_KL_logps = self.kto_forward(self.policy, batch)
            
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, reference_KL_logps = self.kto_forward(self.reference_model, batch)
            
            losses, chosen_rewards, rejected_rewards, KL = self.kto_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                policy_KL_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                reference_KL_logps,
            )
            # GDPO
            losses = losses + CE_losses - policy_logps_belief 
            
            combined_rewards = torch.cat((chosen_rewards.detach(), rejected_rewards.detach()), 0)
            combined_statuses = torch.cat((torch.ones_like(chosen_rewards), torch.zeros_like(rejected_rewards)), 0)

            all_rewards = all_gather_if_needed(combined_rewards.detach(), self.rank, self.world_size)
            all_statuses = all_gather_if_needed(combined_statuses.detach(), self.rank, self.world_size)
            all_KL = all_gather_if_needed(KL, self.rank, self.world_size)
            chosen_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
            rejected_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]

            metrics[f'rewards_{train_test}/chosen'] = all_rewards[chosen_rewards_idx]
            metrics[f'rewards_{train_test}/rejected'] = all_rewards[rejected_rewards_idx]
            metrics[f'rewards_{train_test}/margins'] = torch.Tensor([(all_rewards[chosen_rewards_idx].mean().nan_to_num(0) - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)).item()])
            metrics[f'rewards_{train_test}/KL_estimate'] = all_KL
            # del policy_chosen_logps, policy_rejected_logps, policy_KL_logps, reference_chosen_logps, reference_rejected_logps, reference_KL_logps
            # del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_KL
        CE_losses = all_gather_if_needed(CE_losses.detach(), self.rank, self.world_size)
        metrics[f'KL_losses_{train_test}'] = CE_losses.cpu().numpy().tolist()
        JS_divergence = all_gather_if_needed(JS_divergence.detach(), self.rank, self.world_size)
        metrics[f'JS_divergence_{train_test}'] = JS_divergence.cpu().numpy().tolist()
        CLS_Sum = all_gather_if_needed(cls_sum.detach(), self.rank, self.world_size)
        metrics[f'CLS_Sum_{train_test}'] = CLS_Sum.cpu().numpy().tolist()
        # policy_chosen_logps = all_gather_if_needed(-policy_chosen_logps.detach(), self.rank, self.world_size)
        # metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()
        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()
        return losses.mean(), metrics
    
    def remove_directory_safely(path):
        if dist.is_initialized():
            dist.barrier()  # Synchronize all processes before deletion

        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except FileNotFoundError as e:
                rank0_print(f"Warning: File not found during deletion - {e}")
        else:
            rank0_print(f"Directory {path} not found. Skipping deletion.")

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo', 'mpo', 'cdpo','kto'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None
        
        best_eval_metric = None
        patience_counter = 0
        best_model_path = None
        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                # rank0_print(f'Running evaluation after {self.example_counter} train examples')
                # self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {'dpo', 'ipo','mpo', 'cdpo','kto'}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False, eval_name='eval')

                    for k, v in eval_metrics.items():
                        # print(k, v)
                        all_eval_metrics[k].extend(v)
                # for eval_batch in (tqdm.tqdm(self.eval_batches_0, desc='Computing eval_0 metrics') if self.rank==0 else self.eval_batches_0):
                #     local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                #     with torch.no_grad():
                #         _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False, eval_name='eval_0')

                #     for k, v in eval_metrics.items():
                #         all_eval_metrics[k].extend(v)
                # for eval_batch in (tqdm.tqdm(self.eval_batches_1, desc='Computing eval_1 metrics') if self.rank==0 else self.eval_batches_1):
                #     local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                #     with torch.no_grad():
                #         _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False, eval_name='eval_1')

                #     for k, v in eval_metrics.items():
                #         all_eval_metrics[k].extend(v)
                # for eval_batch in (tqdm.tqdm(self.eval_batches_2, desc='Computing eval_2 metrics') if self.rank==0 else self.eval_batches_2):
                #     local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                #     with torch.no_grad():
                #         _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False, eval_name='eval_2')

                #     for k, v in eval_metrics.items():
                #         all_eval_metrics[k].extend(v)


                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {'dpo', 'ipo','mpo', 'cdpo', 'kto'}:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:                    
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {'dpo', 'ipo','mpo', 'cdpo','kto'}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {'dpo', 'ipo','mpo', 'cdpo','kto'}:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        # output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        # output_dir = os.path.join(self.config.output_dir, f'step-{self.example_counter}')
                        # rank0_print(f'creating checkpoint to write to {output_dir}...')
                        # # customize the save path
                        # self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####
            
                        #### Model Selection: Save the best model based on the evaluation metric
                        current_eval_metric = mean_eval_metrics[self.config.eval_metric]  # Use your desired eval metric
                        if best_eval_metric is None or current_eval_metric < best_eval_metric:
                            # If a better model is found, delete the previous best model
                            if best_model_path is not None and os.path.exists(best_model_path):
                                rank0_print(f'Removing previous best model at {best_model_path}')
                                if self.rank == 0:
                                    shutil.rmtree(best_model_path)


                            best_eval_metric = current_eval_metric
                            best_model_path = os.path.join(self.config.output_dir, f'step-{self.example_counter}')
                            rank0_print(f'New best model with {self.config.eval_metric}={best_eval_metric} at step {self.example_counter}, saving to {best_model_path}')
                            self.save(best_model_path, mean_eval_metrics)
                            patience_counter = 0  # Reset the patience counter
                        else:
                            patience_counter += 1
                            rank0_print(f'No improvement in {self.config.eval_metric} after {patience_counter} evals.')
                        
                        # Early Stopping: Stop training if no improvement after `patience` evaluations
                        if patience_counter >= self.config.patience:
                            rank0_print(f'Early stopping triggered after {patience_counter} evaluations without improvement.')
                            break
            
            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {dir_name} and {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)

class UnpairedPreferenceTrainer(BasicTrainer):
    use_reference_model = True

    """A trainer for any loss that doesn't use paired preference, like KTO."""
    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], use_cache: bool=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        """Run the given model on the given batch of inputs.
        
        Returns:
            chosen_logps: log probabilities of chosen examples 
            rejected_logps: log probabilities of rejected examples
            use_cache: if true, expecte to get cached logprobs from the model
        """
        with self.accelerator.autocast():
            if use_cache:
                all_logps = model(batch['target_combined_input_ids']).to(self.policy_dtype)
            else:
                all_logits = model(
                    batch['target_combined_input_ids'], 
                    attention_mask=batch['target_combined_attention_mask'],
                ).logits.to(self.policy_dtype)
            
                all_logps = self.get_batch_logps(all_logits, batch['target_labels'])

        assert all_logps.shape[0] == len(batch['status'])
        chosen_idx = [i for i in range(all_logps.shape[0]) if batch['status'][i] == 'chosen']
        rejected_idx = [i for i in range(all_logps.shape[0]) if batch['status'][i] == 'rejected']

        chosen_logps = all_logps[chosen_idx, ...]
        rejected_logps = all_logps[rejected_idx, ...]
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}

        if self.reference_model is None:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps)
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)

        # all_gather treats empty lists/tensors poorly, and empty lists can occur because a batch can contain all chosen or all rejected example
        # therefore, concatenate chosen + rejected rewards before all_gather
        combined_rewards = torch.cat((chosen_rewards.detach(), rejected_rewards.detach()), 0)
        combined_statuses = torch.Tensor([1] * len(chosen_rewards) + [0] * len(rejected_rewards))

        all_rewards = self.accelerator.gather(combined_rewards.detach())
        all_statuses = self.accelerator.gather(combined_statuses.detach())
        chosen_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
        rejected_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]

        metrics[f'rewards_{mode}/chosen'] = all_rewards[chosen_rewards_idx]
        metrics[f'rewards_{mode}/rejected'] = all_rewards[rejected_rewards_idx]
        metrics[f'rewards_{mode}/margins'] = torch.Tensor([(all_rewards[chosen_rewards_idx].mean().nan_to_num(0) - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)).item()])
        metrics[f'loss/{mode}'] = self.accelerator.gather(losses.mean().detach()).mean()

        del policy_chosen_logps, policy_rejected_logps
        del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_devices_losses

        if self.reference_model:
            del reference_chosen_logps, reference_rejected_logps

        return losses.sum(), metrics

class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo','mpo', 'cdpo','kto'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()
        
class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        
        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo','mpo', 'cdpo','kto'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        
class KTOTrainer(UnpairedPreferenceTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities.

        If generation y ~ p_desirable, we have the 'desirable' loss:
            L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - KL(p_policy || p_reference)))
        If generation y ~ p_undesirable, we have the 'undesirable' loss:
            L(x, y) := 1 - sigmoid(beta * (KL(p_policy || p_reference) - [log p_policy(y|x) - log p_reference(y|x)]))

        The desirable losses are weighed by config.loss.desirable_weight.
        The undesirable losses are weighed by config.loss.undesirable_weight.
        This should be used to address imbalances in the ratio of desirable:undesirable examples respectively.

        The KL term is estimated by matching x with unrelated outputs y', then calculating the average log ratio
        log p_policy(y'|x) - log p_reference(y'|x). Doing so avoids the requirement that there be equal numbers of 
        desirable and undesirable examples in the microbatch.
        """
        KL_rewards = policy_KL_logps.sum(-1) - reference_KL_logps.sum(-1)
        # take mean of the KL estimates across all devices in this step
        KL = self.accelerator.gather(KL_rewards.detach()).mean().clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_rewards = (policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1))
            chosen_losses = self.config.loss.desirable_weight * (1 - F.sigmoid(self.config.loss.beta * (chosen_rewards - KL)))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(self.policy_dtype)
            chosen_rewards = torch.Tensor([]).to(self.policy_dtype)
        
        if policy_rejected_logps.shape[0] != 0:
            rejected_rewards = (policy_rejected_logps.sum(-1) - reference_rejected_logps.sum(-1))
            rejected_losses = self.config.loss.undesirable_weight * (1 - F.sigmoid(self.config.loss.beta * (KL - rejected_rewards)))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(self.policy_dtype)
            rejected_rewards = torch.Tensor([]).to(self.policy_dtype)
        print("devices of losses", chosen_losses.device, rejected_losses.device)
        losses = torch.cat((chosen_losses, rejected_losses), 0)

        return losses, chosen_rewards.detach(), rejected_rewards.detach(), KL.detach()
    
    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], use_cache: bool=False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs.
        
        Args:
            - model: the model to use for the forward pass
            - batch: the microbatch (should have the input ids, attention mask, and labels)
            - use_cache: if true, can get cached logprobs instead

        Returns:
            chosen_logps: log probabilities of chosen examples
            rejected_logps: log probabilities of rejected examples
            KL_logps: log probabilities of the unmatched y'|x (used to estimate the KL divergence between policy and reference)
        """
        batch = exclude_belief_from_batch(batch, self.tokenizer)
        with self.accelerator.autocast():
            with torch.no_grad():
                if use_cache:
                    KL_logps = model(batch[f'KL_combined_input_ids']).to(self.policy_dtype)
                else:
                    KL_logits = model(
                        batch[f'KL_combined_input_ids'],
                        attention_mask=batch[f'KL_combined_attention_mask']
                    ).logits.to(self.policy_dtype)

                    KL_logps = self.get_batch_logps(KL_logits, batch[f'KL_labels'])

            if use_cache:
                target_logps = model(batch[f'target_combined_input_ids']).to(self.policy_dtype)
            else:
                target_logits = model(
                    batch[f'target_combined_input_ids'],
                    attention_mask=batch[f'target_combined_attention_mask']
                ).logits.to(self.policy_dtype)

                target_logps = self.get_batch_logps(target_logits, batch[f'target_labels'])

        assert target_logps.shape[0] == len(batch['status'])
        chosen_idx = [i for i in range(target_logps.shape[0]) if batch['status'][i] == 'chosen']
        rejected_idx = [i for i in range(target_logps.shape[0]) if batch['status'][i] == 'rejected']
        chosen_logps = target_logps[chosen_idx, ...]
        rejected_logps = target_logps[rejected_idx, ...]

        return chosen_logps, rejected_logps, KL_logps
    
    # GDPO version
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}
        policy_chosen_logits = self.policy(
                batch['target_combined_input_ids'], 
                attention_mask=batch['target_combined_attention_mask'],
            ).logits.to(self.policy_dtype)
        CE_losses, JS_divergence, cls_sum = _CE_loss(batch, policy_chosen_logits, self.tokenizer)
        policy_logps_belief = _get_batch_logps_belief(batch, policy_chosen_logits, batch['target_labels'], self.tokenizer, average_log_prob=True)
        # policy_logps_belief = self.get_batch_logps(batch, policy_chosen_logits, batch['target_labels'])

        policy_chosen_logps, policy_rejected_logps, policy_KL_logps = self.forward(self.policy, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, reference_KL_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)

        
        losses, chosen_rewards, rejected_rewards, KL = self.loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps,
        )

        combined_rewards = torch.cat((chosen_rewards.detach(), rejected_rewards.detach()), 0)
        combined_statuses = torch.Tensor([1] * len(chosen_rewards) + [0] * len(rejected_rewards))

        all_rewards = self.accelerator.gather(combined_rewards)
        all_statuses = self.accelerator.gather(combined_statuses)
        all_KL = self.accelerator.gather(KL)
        chosen_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
        rejected_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]

        metrics[f'rewards_{mode}/chosen'] = all_rewards[chosen_rewards_idx]
        metrics[f'rewards_{mode}/rejected'] = all_rewards[rejected_rewards_idx]
        metrics[f'rewards_{mode}/margins'] = torch.Tensor([(all_rewards[chosen_rewards_idx].mean().nan_to_num(0) - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)).item()])
        metrics[f'rewards_{mode}/KL_estimate'] = all_KL

        losses = CE_losses - policy_logps_belief

        metrics[f'loss/{mode}'] = self.accelerator.gather(losses.mean().detach()).mean()
        metrics[f'CE_losses/{mode}'] = self.accelerator.gather(CE_losses.mean().detach()).mean()
        metrics[f'JS_divergence/{mode}'] = self.accelerator.gather(JS_divergence.mean().detach()).mean()
        metrics[f'cls_sum/{mode}'] = self.accelerator.gather(cls_sum.mean().detach()).mean()
        metrics[f'policy_logps_belief/{mode}'] = self.accelerator.gather(policy_logps_belief.mean().detach()).mean()

        del policy_chosen_logps, policy_rejected_logps, policy_KL_logps, reference_chosen_logps, reference_rejected_logps, reference_KL_logps
        del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_KL

        return losses.sum(), metrics
    
    # KTO version

    # def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str='train'):
    #     """Compute the loss and other metrics for the given batch of inputs."""
    #     metrics = {}

    #     policy_chosen_logps, policy_rejected_logps, policy_KL_logps = self.forward(self.policy, batch)
    #     with torch.no_grad():
    #         reference_chosen_logps, reference_rejected_logps, reference_KL_logps = self.forward(self.reference_model, batch, use_cache=self.config.cache_reference_logprobs)
    #         policy_chosen_logits = self.policy(
    #             batch['target_combined_input_ids'], 
    #             attention_mask=batch['target_combined_attention_mask'],
    #         ).logits.to(self.policy_dtype)
    #         CE_losses, JS_divergence, cls_sum = _CE_loss(batch, policy_chosen_logits, self.tokenizer)

        
    #     losses, chosen_rewards, rejected_rewards, KL = self.loss(
    #         policy_chosen_logps,
    #         policy_rejected_logps,
    #         policy_KL_logps,
    #         reference_chosen_logps,
    #         reference_rejected_logps,
    #         reference_KL_logps,
    #     )

    #     combined_rewards = torch.cat((chosen_rewards.detach(), rejected_rewards.detach()), 0)
    #     combined_statuses = torch.Tensor([1] * len(chosen_rewards) + [0] * len(rejected_rewards))

    #     all_rewards = self.accelerator.gather(combined_rewards)
    #     all_statuses = self.accelerator.gather(combined_statuses)
    #     all_KL = self.accelerator.gather(KL)
    #     chosen_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
    #     rejected_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]

    #     metrics[f'rewards_{mode}/chosen'] = all_rewards[chosen_rewards_idx]
    #     metrics[f'rewards_{mode}/rejected'] = all_rewards[rejected_rewards_idx]
    #     metrics[f'rewards_{mode}/margins'] = torch.Tensor([(all_rewards[chosen_rewards_idx].mean().nan_to_num(0) - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)).item()])
    #     metrics[f'rewards_{mode}/KL_estimate'] = all_KL
    #     metrics[f'loss/{mode}'] = self.accelerator.gather(losses.mean().detach()).mean()
    #     metrics[f'CE_losses/{mode}'] = self.accelerator.gather(CE_losses.mean().detach()).mean()
    #     metrics[f'JS_divergence/{mode}'] = self.accelerator.gather(JS_divergence.mean().detach()).mean()
    #     metrics[f'cls_sum/{mode}'] = self.accelerator.gather(cls_sum.mean().detach()).mean()
    #     del policy_chosen_logps, policy_rejected_logps, policy_KL_logps, reference_chosen_logps, reference_rejected_logps, reference_KL_logps
    #     del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_KL

    #     return losses.sum(), metrics
