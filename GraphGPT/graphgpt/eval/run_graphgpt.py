import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)

import argparse
import copy
import json
import os
import os.path as osp
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Sequence

import pandas as pd
import ray
import requests
import torch
import transformers
from graphgpt.conversation import SeparatorStyle, conv_templates
from graphgpt.model import *
from graphgpt.model.GraphLlama_pl import GraphGPT_pl
from graphgpt.model.utils import KeywordsStoppingCriteria
from graphgpt.train.train_light import DataArguments, ModelArguments, TrainingArguments
from graphgpt.utils import disable_torch_init
from PIL import Image
from torch_geometric.data import Data
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    StoppingCriteria,
)

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


def load_graph(graph, bert_tokenizer, bert_model): 
    # graph_data_all = torch.load(graph_data_path)
    # graph_dict = instruct_item['graph']
    graph_edge_index = torch.Tensor(copy.deepcopy(graph['connectivity'])).long()
    graph_node_list = copy.deepcopy(graph['nodes'])
    # target_node = copy.deepcopy(graph_dict['node_idx'])
    # graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    raw_node_attri = [str(e) for e in graph_node_list]
    # graph_node_rep = graph_data_all[graph_type].x[graph_node_list] ## 
    graph_node_rep = bert_tokenizer(
            raw_node_attri,
            return_tensors="pt",
            padding="longest",
            max_length=bert_tokenizer.model_max_length,
            truncation=True
        )
    for key, value in graph_node_rep.items():
            graph_node_rep[key] = value.to(bert_model.device)

    graph_node_rep = bert_model(**graph_node_rep).pooler_output
    cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size
    # graph_node_rep = graph_node_rep.to(compute_dtype)
    graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index)

    return {
        'graph_data': graph_ret, 
        'graph_token_len': cur_token_len
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    graph_data_all = pd.read_json(args.graph_data_path, lines=True)
    # prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(0, len(prompt_file), chunk_size))
    # split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus: 
        split_list.append(len(prompt_file))
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = len(prompt_file)
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')

    if osp.exists(args.output_res_path) is False: 
        os.makedirs(args.output_res_path, exist_ok=True)
    
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split, graph_data_all.iloc[start_idx:end_idx]
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx, graph_pd):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)


    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    # print('start loading')
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # print('finish loading')

    print('start initiate model')
    # print('start loading')
    model_args = ModelArguments(
        model_name_or_path=args.tokenizer_path,
        version="v1",
        graph_tower='clip_gt_arxiv',
        tune_graph_mlp_adapter=True,
        graph_select_layer=-2,
        use_graph_start_end=True,
        freeze_backbone=True,
        num_query_token=args.num_query_tokens,
    )
    data_args = DataArguments(
        # data_path='/data/LPJ/ICML25/graphgpt_dataset/gpt_dataset_construction/rtlcoder_gpt4_v1/import_for_graphgpt/conversations.json',
        # graph_data_path='/data/LPJ/ICML25/graphgpt_dataset/gpt_dataset_construction/rtlcoder_gpt4_v1/import_for_graphgpt/graph.jsonl',
        lazy_preprocess=True,
        bert_path=args.bert_path,
        bert_gpu=3,
        # bert_tokenizer_max_length=15,
        graph_content="./arxiv_ti_ab.json",
        bert_tokenizer_max_length=args.bert_tokenizer_max_length,

    )
    train_args = TrainingArguments(
        bf16=args.bf16,
        fp16=args.f16,
        output_dir='/data/LPJ/ICML25/GraphGPT/checkpoints/pretraining_stage/v0',
        num_train_epochs=3,
        model_max_length=args.model_max_length,
        # gpus='cpu',
        gpus='0',
        lora_enable=args.lora_enable,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    print('start initiate tokenizer from {}'.format(args.tokenizer_path))
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path, model_max_length=train_args.model_max_length, padding_side="right")
    # print('set pad token to eos token')
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    print('finish initiate tokenizer')
    if args.load_from_ckpt:
    # test_model = GraphGPT_pl(training_args=train_args, model_args=model_args, data_args=data_args, tokenizer=tokenizer)
    # ckpt = torch.load(args.model_name, map_location='cpu')
        model = GraphGPT_pl.load_from_checkpoint(checkpoint_path=args.model_name
                                        #  ,training_args=train_args, model_args=model_args, data_args=data_args, tokenizer=tokenizer)
                                         ,training_args=train_args, model_args=model_args, data_args=data_args, tokenizer=tokenizer, map_location='cpu')
    else:
        model_args.pretrain_graph_mlp_adapter = args.pretrain_graph_mlp_adapter
        model_args.pretrain_input_embedding_path = args.pretrain_input_embedding_path
        model = GraphGPT_pl(training_args=train_args, model_args=model_args, data_args=data_args, tokenizer=tokenizer)
    

    if args.lora_enable:
        model = model.model.merge_and_unload()
    else:
        model = model.model

    compute_dtype = (torch.float16 if train_args.fp16 else (torch.bfloat16 if train_args.bf16 else torch.float32))
    model = model.to(dtype=compute_dtype)
    model = model.cuda()
    # model = model.model.merge_and_unload().cuda()
    # model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    
    print('finish loading')

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    if not args.use_trained_gnn:
        print('not use trained gnn, use pretrained gnn!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    graph_tower = model.get_model().graph_tower
    
    # TODO: add graph tower
    # if graph_tower.device.type == 'meta':
    #     print('meta')
    # clip_graph, args_graph= load_model_pretrained(CLIP, '/data/LPJ/ICML25/GraphGPT/clip_gt_arxiv')
    # # clip_graph, args_graph= load_model_pretrained(CLIP, './clip_gt_arxiv')
    # graph_tower = graph_transformer(args_graph)
    # graph_tower = transfer_param_tograph(clip_graph, graph_tower)
    # # graph_tower = graph_tower.to(torch.float32)
    # # model.get_model().graph_tower = graph_tower
    # model.get_model().graph_tower = graph_tower.cuda()
    # # else:
    # #     print('other')
    # # print(next(graph_tower.parameters()).dtype)
    # # graph_tower.to(device='cpu', dtype=torch.float32)
    # graph_tower.to(device='cuda', dtype=compute_dtype)
    # graph_config = graph_tower.config
    # graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    # graph_config.use_graph_start_end = use_graph_start_end
    # if use_graph_start_end:
    #     graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # TODO: add graph token len

    # res_data = []
    print(f'total: {len(prompt_file)}')
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_path,
        model_max_length=args.bert_tokenizer_max_length,
        padding_side="right")
    
    bert_model = BertModel.from_pretrained(args.bert_path, torch_dtype=torch.bfloat16).to('cpu')
    # bert_model = BertModel.from_pretrained(args.bert_path, torch_dtype=torch.bfloat16).to('cuda')
    # model = model.float()
    # bert_model = bert_model.float()
    for idx, (instruct_item, (graph_index, graph)) in tqdm(enumerate(zip(prompt_file, graph_pd.iterrows())), total=len(prompt_file)):
    # for idx, (instruct_item, graph) in tqdm(enumerate((prompt_file, graph_pd.iterrows()))):
        # instruct_item = prompt_file[0]
        # if idx >= 3: 
        #     break
        for i in range(args.n_pass_k):

            res_data = []

            graph_dict = load_graph( graph=graph, bert_tokenizer=bert_tokenizer, bert_model=bert_model)
            # graph_token_len = graph_dict['graph_token_len']
            graph_token_len = args.num_query_tokens
            graph_data = graph_dict['graph_data']

            qs = instruct_item["conversations"][0]["value"]
            # if use_graph_start_end:
            #     qs = qs + '\n' + DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len + DEFAULT_G_END_TOKEN
            # else:
            #     qs = qs + '\n' + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len

            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)

            # if "v1" in args.model_name.lower():
            #     conv_mode = "graphchat_v1"
            # else: 
            #     raise ValueError('Don\'t support this model')
            conv_mode = "graphchat_v1"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            

            # input_ids = torch.as_tensor(inputs.input_ids)
            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # graph_data.graph_node = graph_data.graph_node.to(torch.float32)
            graph_data.graph_node = graph_data.graph_node.to(compute_dtype)
            # graph_data.edge_index = graph_data.edge_index.to(torch.float16)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    # graph_data=graph_data,
                    graph_data=graph_data.cuda(),
                    # do_sample=False,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.model_max_length,
                    stopping_criteria=[stopping_criteria])
            # torch.cuda.empty_cache()
            # print("==============================================================")
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            # print(outputs)

            res_data.append({"task_id": instruct_item["task_id"], "response": outputs}.copy())
            # res_data.append({"id": instruct_item["id"], "node_idx": instruct_item["graph"]["node_idx"], "res": outputs}.copy())
            # current_time = datetime.now()
            # formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            directory = args.output_res_path
            # directory = osp.join(args.output_res_path, 'test{}/'.format(i))
            os.makedirs(directory, exist_ok=True)
            file_path = osp.join(directory, '{}.jsonl'.format(args.output_file_name))
            with open(file_path, "a") as fout:
            # with open(osp.join(args.output_res_path, 'hiverilog_test_res_{}.json'.format(formatted_time)), "w") as fout:
            # with open(osp.join(args.output_res_path, 'arxiv_test_res_{}_{}.json'.format(start_idx, end_idx)), "w") as fout:
                # json.dump(res_data, fout, )
                if isinstance(res_data, dict):
                    json.dump(res_data, fout)
                    fout.write('\n')  # 添加换行符

                # 如果 res_data 是一个列表，逐个写入
                elif isinstance(res_data, list):
                    for entry in res_data:
                        json.dump(entry, fout)
                        fout.write('\n')  # 添加换行符
            torch.cuda.empty_cache()
            

    return res_data
    # with open(args.output_res_path, "w") as fout:
    #     json.dump(res_data, fout, indent=4)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
    parser.add_argument("--tokenizer_path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--output_file_name", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--bert_path", type=str, default='/data/LPJ/uncased_L-12_H-128_A-2')
    parser.add_argument("--bert_tokenizer_max_length", type=int, default=15)
    parser.add_argument("--model_max_length", type=int, default=3072)
    parser.add_argument("--bf16", type=str2bool, default=False)
    parser.add_argument("--f16", type=str2bool, default=False)
    parser.add_argument("--use_trained_gnn", type=str2bool, default=False)
    parser.add_argument("--n_pass_k", type=int, default=10)
    parser.add_argument("--lora_enable", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--num_query_tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--load_from_ckpt", type=str2bool, default=False)
    parser.add_argument("--pretrain_graph_mlp_adapter", type=str, default=None)
    parser.add_argument("--pretrain_input_embedding_path", type=str, default=None)
    
    # parser.add_argument("--start_id", type=int, default=0)
    # parser.add_argument("--end_id", type=int, default=20567)

    args = parser.parse_args()
    # eval_model(args)
    # print("++++++++++++++++++++++++++++++++", args.lora_enable)
    # ray.init()
    ray.init(local_mode=True)
    run_eval(args, args.num_gpus)


# protobuf             4.22.3