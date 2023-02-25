from tokenizer_utils import encode_text, cls_id, mask_id, pad_id, sop_id, eos_id
import argparse
import json
import numpy as np

def create_single_from_seq_pair(input_str: str, output_str: str, cls_label:int, pad_length: int):
    input_seq = encode_text(input_str)
    output_seq = encode_text(output_str)
    if mask_id not in input_seq:
        source_tokens = [cls_id] + input_seq + [mask_id]
    else:
        source_tokens = [cls_id] + input_seq
    mask_pos = source_tokens.index(mask_id)
    input_ids = source_tokens + [sop_id] + output_seq
    target_ids = source_tokens + output_seq + [eos_id]
    context_length = len(source_tokens)
    full_length = len(input_ids)
    if full_length > pad_length:
        return None
    input_ids = input_ids + [pad_id] * (pad_length - full_length)
    target_ids = target_ids + [pad_id] * (pad_length - full_length)
    position_ids = list(range(context_length)) + \
        [mask_pos] * (pad_length - context_length)
    block_position_ids = [0] * context_length + \
        list(range(1, 1+(pad_length - context_length)))
    output = input_ids + target_ids + position_ids + \
        block_position_ids + [context_length, full_length, cls_label]
    return output

def jsonl_iter(infile_path):
    with open(infile_path, 'r') as infile:
        for line in infile:
            item = json.loads(line)
            yield (item['input_str'], item['output_str'], item['label'])

def run(in_filepath, out_path, pad_length,):
    cnt, t_cnt = 0, 0
    outf = open(out_path, 'wb')
    data_iter = jsonl_iter(in_filepath)
    for (input_str, output_str, cls_label) in data_iter:
        t_cnt += 1
        entry = create_single_from_seq_pair(input_str, output_str, cls_label, pad_length)
        if entry:
            cnt += 1
            outf.write(np.array(entry, dtype=np.int32).tobytes())
    outf.close()
    print(f"TotalNum:{t_cnt}, KeepNum:{cnt}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--pad-length', type=int, default=128)
    args = parser.parse_args()
    run(args.in_path, args.out_path, args.pad_length)