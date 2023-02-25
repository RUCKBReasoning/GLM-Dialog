
import torch
import torch.nn as nn
import torch.nn.functional as F
from SwissArmyTransformer.model.base_model import BaseMixin, BaseModel, non_conflict
from SwissArmyTransformer.model.transformer import standard_attention
from SwissArmyTransformer import mpu
import math

from SwissArmyTransformer import mpu
from SwissArmyTransformer.transformer_defaults import standard_attention
from SwissArmyTransformer.mpu.utils import split_tensor_along_last_dim, divide
from SwissArmyTransformer.mpu.layers import ColumnParallelLinear
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin
from SwissArmyTransformer.model.position_embedding import RotaryEmbedding
from SwissArmyTransformer.model.position_embedding import apply_rotary_pos_emb_index
from SwissArmyTransformer.mpu.utils import divide
from SwissArmyTransformer.mpu.initialize import get_model_parallel_world_size
from torch import Tensor

import torch.nn.functional as F

class XGLMFinalMixin(BaseMixin):
    def __init__(self, hidden_size, class_num=2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, class_num)

    def final_forward(self, logits, **kwargs):
        token_logits = F.linear(logits, self.transformer.word_embeddings.weight)
        cls_logits = self.dense(F.relu(logits))
        return (token_logits, cls_logits)

class BlockPositionEmbeddingMixin(BaseMixin):
    def __init__(self, max_sequence_length, hidden_size, init_method_std=0.02):
        super(BlockPositionEmbeddingMixin, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.block_position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
    
    def position_embedding_forward(self, position_ids, **kwargs):
        position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.transformer.position_embeddings(position_ids)
        block_position_embeddings = self.block_position_embeddings(block_position_ids)
        return position_embeddings + block_position_embeddings

class XGLMModel(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('block_position_embedding', 
            BlockPositionEmbeddingMixin(args.max_sequence_length, args.hidden_size)
        )
        self.add_mixin('cls_mixin',XGLMFinalMixin(args.hidden_size))
    
    @classmethod
    def add_model_specific_args(cls, parser):
        """Arguments for GLM"""
        group = parser.add_argument_group('GLM', 'GLM Configurations')
        group.add_argument('--tokenizer-model-type', type=str,
                       default=None,
                       help="Model type to use for sentencepiece tokenization \
                           (one of ['bpe', 'char', 'unigram', 'word']) or \
                           bert vocab to use for BertWordPieceTokenizer (one of \
                           ['bert-large-uncased', 'bert-large-cased', etc.])")
        return parser