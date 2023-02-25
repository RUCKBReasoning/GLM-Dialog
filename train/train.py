from torch.utils.data import Dataset
import os
import sys
import math
import random
import torch
import argparse
import numpy as np

from SwissArmyTransformer.training import initialize_distributed, set_random_seed
from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import BinaryDataset
from xglm_modeling import XGLMModel


def freeze_part_model(model, keep_layer_num=8):
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.transformer.layers[-keep_layer_num:]:
        param.requires_grad_(True)
    return model
   
def setup_model(args):
    model, args = XGLMModel.from_pretrained(args, args.ckpt_path)
    # If you dont have enough memory, it is a choice
    # model = freeze_part_model(model) 
    return model, args


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['tokens', 'labels', 'loss_mask', 'position_ids', 'attention_mask', 'cls_label']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()

    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['tokens'].long()
    labels = data_b['labels'].long()
    loss_mask = data_b['loss_mask'].long()
    attention_mask = data_b['attention_mask'].float()
    position_ids = data_b['position_ids'].long()
    cls_label = data_b['cls_label'].long()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids, cls_label


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids,cls_label = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    # Forward model.
    (logits, cls_logits), *mems = model(tokens, position_ids, attention_mask)

    # cls_loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    cls_loss = loss_fct(cls_logits[:,0,:].float(), cls_label.view(-1))

    # token_loss
    losses = mpu.vocab_parallel_cross_entropy(
        logits.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / loss_mask.sum()

    # add_it
    loss = loss + cls_loss

    return loss, {}


def create_dataset_function(path, args):
    sample_length = args.my_sample_length
    layout = [0, sample_length, sample_length+sample_length, sample_length+sample_length +
              sample_length*2, sample_length+sample_length+sample_length*2+2, sample_length+sample_length+sample_length*2+2+1]  # FIXME

    def process_fn(row):
        row = row.astype(np.int64)
        codes = [row[layout[i-1]:layout[i]] for i in range(1, len(layout))]
        tokens, labels, position_ids, att_flags, cls_label = codes
        tokens = tokens.reshape(sample_length)
        labels = labels.reshape(sample_length)
        cls_label = cls_label.reshape(1)
        position_ids = position_ids.reshape(2, sample_length)
        attention_mask = np.ones(
            (1, sample_length, sample_length), dtype=np.int64)
        context_length, full_length = att_flags
        attention_mask = np.tril(attention_mask)
        attention_mask[:, :, :context_length] = 1
        attention_mask[:, full_length:, :] = 0
        loss_mask = np.zeros_like(tokens)
        loss_mask[context_length:full_length] = 1
        return dict(tokens=tokens, labels=labels, loss_mask=loss_mask, position_ids=position_ids, attention_mask=attention_mask, cls_label=cls_label)
    return BinaryDataset(path, process_fn, length_per_sample=layout[-1], dtype=np.int32)

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--ckpt-path', type=str, default=None)
    py_parser.add_argument('--my-sample-length', type=int, required=True)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    print('args.deepspeed', args.deepspeed)
    model, args = setup_model(args)
    training_main(args, model_cls=model, forward_step_function=forward_step,
                  create_dataset_function=create_dataset_function)
