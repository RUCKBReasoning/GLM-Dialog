from xkdai.generation_model import GenerationModel
from xkdai.chatbot import Chatbot
from xkdai.search_engine import BingSearcher
from pymongo import MongoClient
import json
from flask import (
    Flask,
    request,
    jsonify
)


def load_glm_model_and_tokenizer(my_ckpt_path):
    import argparse
    from SwissArmyTransformer.model import GLMModel
    from SwissArmyTransformer.model.mixins import CachedAutoregressiveMixin
    from SwissArmyTransformer import get_args, get_tokenizer, AutoModel
    from SwissArmyTransformer.training import initialize_distributed
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--my_post_to_cuda', type=bool, default=True)
    GLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args.fp16 = True
    args = argparse.Namespace(**vars(args), **vars(known))
    initialize_distributed(args)
    model, args = AutoModel.from_pretrained(args, my_ckpt_path)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    model.transformer.parallel_output = False
    model = model.cpu()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = get_tokenizer(args)
    return model, tokenizer


APP_CONFIG = {
    'device': 'cuda',
    'ckpt_path': '/data/KDaiCKPTs/glm-large-zh',
    'mongo': {
        'host': 'localhost',
        'port': 27017,
        'username': '',
        'password': ''
    },
    'chatbot': {
        'query_limit': 10,
        'response_limit': 100,
    },
    'decode': {
        'input_max_length': 1024,
        'temperature': 0.9,
        'top_k': 40,
    },
    'search_engine': {
        'use_cache': False,
        'timeout': 2,
        'max_tries': 1,
    },
    'flask': {
        'port': 5007,
    },
}


def main():
    # set up mongo_db
    mongo_config = APP_CONFIG['mongo']
    mongo_client = MongoClient(mongo_config['host'], mongo_config['port'],
                               username=mongo_config['username'], password=mongo_config['password'])
    mongo_client.list_database_names()  # make sure it is setup
    mongo_db = mongo_client['xkdai']
    # set up GenerationModel
    glm_model, tokenizer = load_glm_model_and_tokenizer(
        APP_CONFIG['ckpt_path'])
    generation_model = GenerationModel(
        tokenizer, glm_model, APP_CONFIG['device'], APP_CONFIG['decode']['input_max_length'], APP_CONFIG['decode']['temperature'], APP_CONFIG['decode']['top_k'])
    # set up BingSearcher
    search_engine_config = APP_CONFIG['search_engine']
    searcher_db = mongo_db if search_engine_config['use_cache'] else None
    searcher = BingSearcher(
        search_engine_config['timeout'], search_engine_config['max_tries'], searcher_db)
    # set up chatbot
    chatbot_config = APP_CONFIG['chatbot']
    chatbot = Chatbot(generation_model, searcher, mongo_db,
                      chatbot_config['query_limit'], chatbot_config['response_limit'])
    # wrap the chatbot based on Flask
    app = Flask(__name__)

    @app.route('/generate', methods=["POST"])
    def process():
        input_ = json.loads(request.data)
        utters = input_['content']
        response = chatbot.chat(utters)
        return jsonify(dict(response=response))
    app.run(host='0.0.0.0', port=APP_CONFIG['flask']['port'])


if __name__ == '__main__':
    main()
