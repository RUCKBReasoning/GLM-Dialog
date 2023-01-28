# -*- encoding: utf-8 -*-
'''
@File    :   inference_glm.py
@Time    :   2021/10/22 19:41:58
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
'''

from SwissArmyTransformer.model.official import GLMModel
from SwissArmyTransformer import AutoModel
from functools import partial
import os
import sys
import random
import json
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import argparse
import stat
from functools import partial

from SwissArmyTransformer import mpu, get_args, get_tokenizer

from SwissArmyTransformer.training import initialize_distributed, set_random_seed

from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.model.mixins import CachedAutoregressiveMixin
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence, evaluate_perplexity
from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually


def get_masks_and_position_ids_glm(seq, mask_position, context_length):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(
        2, len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[0, :context_length])
    position_ids[0, context_length:] = mask_position
    torch.arange(1, len(seq) - context_length + 1,
                 out=position_ids[1, context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids


def main(model, args):
    tokenizer = get_tokenizer(args)
    from pprint import pprint
    pprint(args)
    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    set_random_seed(args.seed)
    model.eval()

    end_tokens = [tokenizer.get_command(
        'eop').Id, tokenizer.get_command('eos').Id]
    # define function for each query
    if args.sampling_strategy == 'BaseStrategy':
        strategy = BaseStrategy(
            temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens)
    elif args.sampling_strategy == 'BeamSearchStrategy':
        strategy = BeamSearchStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True,
                                      end_tokens=end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size, min_tgt_length=args.min_tgt_length)
    else:
        raise ValueError(f'unknown strategy {args.sampling_strategy}')

    def process(raw_text):
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        # add MASK
        generation_mask = '[gMASK]' if args.task_mask else '[MASK]'
        if 'MASK]' not in raw_text:
            raw_text += ' ' + generation_mask
        seq = tokenizer.EncodeAsIds(raw_text).tokenization
        print('raw_text encoded as', seq)
        seq = [tokenizer.get_command('ENC').Id] + seq
        if not raw_text.endswith('MASK]'):
            seq = seq + [tokenizer.get_command('eos').Id]
        print('raw text: {}\n'.format(raw_text))
        if len(seq) > args.max_sequence_length:
            raise ValueError('text too long.')

        # generation
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = [seq]
        # continually detect the first mark position
        while True:
            seq = output_list[0]  # TODO find the best one
            # detect
            mask_tokens = ['MASK', 'sMASK',
                           'gMASK'] if args.task_mask else ['MASK']
            mask_tokens = [tokenizer.get_command(
                token).Id for token in mask_tokens]
            mask_position = len(seq)
            for token in mask_tokens:
                try:
                    mask_position = min(mask_position, seq.index(token))
                except ValueError:
                    pass
            if mask_position == len(seq):
                break

            get_func = partial(get_masks_and_position_ids_glm,
                               mask_position=mask_position, context_length=len(seq))
            output_list = []
            for tim in range(max(args.batch_size // mbz, 1)):
                input_seq = torch.cuda.LongTensor(
                    seq +
                    [tokenizer.get_command('sop').Id] + [-1] *
                    (args.out_seq_length - len(seq) - 1),
                    device=args.device)
                output = filling_sequence(model, input_seq,
                                          batch_size=min(args.batch_size, mbz),
                                          strategy=strategy,
                                          log_attention_weights=None,
                                          get_masks_and_position_ids=get_func
                                          )[0]  # we don't use mems, fill back
                if isinstance(output, torch.Tensor):  # different strategies
                    output = list(output)

                output_list.extend(output)

            # clip -1s and fill back generated things into seq
            for i in range(len(output_list)):
                output = output_list[i].tolist()
                try:
                    unfinished = output.index(-1)
                except ValueError:
                    unfinished = len(output)
                if output[unfinished - 1] in end_tokens:
                    unfinished -= 1
                bog = output.index(tokenizer.get_command('sop').Id)
                output_list[i] = output[:mask_position] + \
                    output[bog + 1:unfinished] + output[mask_position + 1:bog]

        # decoding
        txts = []
        for seq in output_list:
            decode_tokens = tokenizer.DecodeIds(seq)
            txts.append(decode_tokens)

        # save
        if args.with_id:
            full_path = os.path.join(args.output_path, query_id + '.txt')
        else:
            prefix = raw_text.replace('/', '')[:20]
            full_path = timed_name(prefix, '.txt', args.output_path)
            print(txts[0])  # print the first.
            return txts[0]

    os.makedirs(args.output_path, exist_ok=True)
    # generate_continually(process, args.input_source)
    with open(args.input_source) as fin:
        inputs = fin.readlines()
        inputs = [json.loads(line) for line in inputs]
    with open(os.path.join(args.output_path, args.outfile), 'w') as outf:
        for input_str in inputs:
            try:
                pred = process(input_str)
            except:
                continue
            pred_json_item = dict(input_str=input_str, pred=pred)
            print(json.dumps(pred_json_item, ensure_ascii=False), file=outf)


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sampling-strategy', type=str,
                           default='BaseStrategy', help='type name of sampling strategy')
    py_parser.add_argument('--my-ckpt-path', type=str, default='glm-large-zh')
    py_parser.add_argument('--outfile', type=str, default='output_q2.txt')
    GLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.do_train = False

    initialize_distributed(args)
    # build model
    model, args = AutoModel.from_pretrained(args, args.my_ckpt_path)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    model.transformer.parallel_output = False

    with torch.no_grad():
        main(model, args)
