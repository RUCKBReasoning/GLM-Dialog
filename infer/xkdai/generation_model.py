import torch
from functools import partial


class GenerationModel:
    def __init__(self, tokenizer, model, device: str, model_input_max_length: int, temperature: float, top_k: int):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.model_input_max_length = model_input_max_length
        self.temperature = temperature
        self.top_k = top_k
        self.end_tokens = [self.tokenizer.get_command(
            'eop').Id, self.tokenizer.get_command('eos').Id]

    def generate(self, prompt, max_generate_length=100) -> str:
        generation_mask = 'sMASK'
        if generation_mask not in prompt:
            prompt += ' ' + generation_mask

        seq = self.tokenizer.EncodeAsIds(prompt).tokenization
        seq = [self.tokenizer.get_command('ENC').Id] + seq
        if len(seq) > self.model_input_max_length:
            raise ValueError('text too long.')

        mask_tokens = ['MASK', 'sMASK',
                       'gMASK']
        mask_tokens = [self.tokenizer.get_command(
            token).Id for token in mask_tokens]
        mask_position = len(seq)
        for token in mask_tokens:
            try:
                mask_position = min(mask_position, seq.index(token))
            except ValueError:
                pass
        if mask_position == len(seq):
            raise ValueError("No mask token found")
        get_func = partial(GenerationModel.get_masks_and_position_ids_glm,
                           mask_position=mask_position, context_length=len(seq))
        input_seq = torch.cuda.LongTensor(
            seq +
            [self.tokenizer.get_command('sop').Id] + [-1] *
            (max_generate_length),
            device=self.device)
        from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy
        from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
        strategy = BaseStrategy(
            temperature=self.temperature, top_k=self.top_k, end_tokens=self.end_tokens)

        output = filling_sequence(self.model, input_seq,
                                  batch_size=1,
                                  strategy=strategy,
                                  log_attention_weights=None,
                                  get_masks_and_position_ids=get_func
                                  )[0]
        output = output[0].tolist()
        unfinished = output.index(-1) if output.count(-1) else len(output)
        bog = output.index(self.tokenizer.get_command('sop').Id)
        generated = output[bog + 1:unfinished]
        return self.tokenizer.DecodeIds(generated)

    @staticmethod
    def get_masks_and_position_ids_glm(seq, mask_position, context_length):
        tokens = seq.unsqueeze(0)

        attention_mask = torch.ones(
            (1, len(seq), len(seq)), device=tokens.device)
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
