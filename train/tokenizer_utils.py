import numpy as np
from SwissArmyTransformer.tokenization.glm import ChineseSPTokenizer
from typing import List, Tuple


def get_tokenizer(tokenizer_name='glm-10b', tokenizer_type='glm_ChineseSPTokenizer'):
    import argparse
    from SwissArmyTransformer import mpu, get_args, get_tokenizer
    py_parser = argparse.ArgumentParser(add_help=False)
    args = py_parser.parse_args([])
    args.tokenizer_model_type = tokenizer_name
    args.tokenizer_type = tokenizer_type
    args.task_mask = 'true'
    args.block_mask_prob = 0.0
    tokenizer = get_tokenizer(args)
    return tokenizer


tokenizer = get_tokenizer('glm-10b')

cls_id = tokenizer.get_command('ENC').Id
mask_token = 'sMASK'
mask_id = tokenizer.get_command(mask_token).Id
pad_id = tokenizer.get_command('pad').Id
sop_id = tokenizer.get_command('sop').Id
eos_id = tokenizer.get_command('eos').Id

def encode_text(text: str):
    return tokenizer.EncodeAsIds(text).tokenization

def decode_text(ids:List[int]):
    return tokenizer.DecodeIds(ids)

