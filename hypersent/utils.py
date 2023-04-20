import os
import re
# import logging
import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging
from hypersent.models import BertForHyper, RobertaForHyper, BertHyperConfig, RobertaHyperConfig
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, List
from geoopt import PoincareBallExact
from dataclasses import dataclass, field

# logger = logging.getLogger(__name__)
logger = logging.get_logger()
Manifold = PoincareBallExact()

@dataclass
class VisualizeEmbeddingsArguments:
    visualize_embeddings: bool = field(
        default=False,
        metadata={'help': 'Whether to visualize embeddings.'}
    )
    embeddings_path: str = field(
        default=None,
        metadata={'help': 'Path to Hyperbolic Embeddings'}
    )
    dim_len: int = field(
        default=2,
        metadata={
            'help': 'Dimensional of visualization.',
            'choices': [2, 3]
        }
    )
    max_num: int = field(
        default=100, 
        metadata={'help': 'Maximum visualized embeddings number.'}
    )
    save_name: str = field(
        default=None,
        metadata={'help': 'Name of the saved figure.'}
    )

def visualize_embeddings(args: VisualizeEmbeddingsArguments):
    if not args.visualize_embeddings:
        return
    assert args.dim_len == 2 or args.dim_len == 3, f"Only support 2/3 dimensional visualization."
    assert os.path.isfile(args.embeddings_path), f"Need to be a path to embeddings file, but got {args.visualize_embeddings}"
    with open(args.embeddings_path, 'r', encoding='utf8') as fo:
        embeddings = json.load(fo)
        logger.info('Done reading.')
    
    embeddings = torch.tensor(embeddings[: args.max_num])
    if args.dim_len < embeddings.shape[-1]:    
        embeddings = Manifold.logmap0(embeddings, dim=-1)
        embeddings = torch.svd(embeddings)[0][:, :args.dim_len]
        embeddings = Manifold.expmap0(embeddings, dim=-1)
        logger.info('Done SVD.')

        embeddings = embeddings / (embeddings.norm(dim=-1, p=2).max() * 1.01)

    if args.dim_len == 2:
        fig, ax = plt.subplots(figsize=(16, 16))
        theta = np.linspace(0, 2 * np.pi, 200)
        x = 1 * np.cos(theta)
        y = 1 * np.sin(theta)
        ax.plot(x, y, color="black", linewidth=1)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color='darkblue', linewidths=0.5)
        ax.xaxis.set_major_locator(plt.NullLocator()) 
        ax.yaxis.set_major_locator(plt.NullLocator()) 
    else:
        fig = plt.figure(figsize=(16, 16))
        ax = Axes3D(fig)
        ax.scatter3D(embeddings[:, 0], embeddings[:, 1], embeddings[:, 1], color='darkblue', linewidths=0.5)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.zaxis.set_major_locator(plt.NullLocator())
    save_name = args.save_name
    if not save_name:
        save_name = f'{args.dim_len}-dimensional.pdf'
    save_dir = os.path.join('results', 'visualize_embeddings')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, save_name))
    logger.info(f'Done saving, save_path={os.path.join(save_dir, save_name)}')


@dataclass
class ParseLogArguments:
    parse_log: bool = field(
        default=False,
        metadata={'help': 'Whether to parse log.'}
    )

    log_path: str = field(
        default=None,
        metadata={'help': 'The path of the log to be parsed.'}
    )

def parse_log(args: ParseLogArguments):
    if not args.parse_log:
        return

    with open(args.log_path, 'r', encoding='utf8') as fi:
        content = fi.read()
    
    step_p = re.compile(r'Total optimization steps = (\d+)')
    loss_p = re.compile(r"'(loss(?:_all|\d))': ([\d\.]{8})")
    metric_p = re.compile(r"'eval_stsb_spearman': ([\d\.]{8})\d+.*?'epoch': ([\d\.]+)")

    step = int(step_p.findall(content)[0])

    losses = loss_p.findall(content)
    loss_type = set(loss_type for loss_type, _ in losses)
    loss_type_num = len(loss_type)
    loss_res = []
    for i in range(len(losses) // loss_type_num):
        item = {'step': i * 125 + 1} 
        for j in range(loss_type_num):
            item[losses[i * loss_type_num + j][0]] = float(losses[i * loss_type_num + j][1])
        loss_res.append(item)
    
    keys = ['step']
    keys.extend(loss_type)
    print('\t'.join(keys))
    for item in loss_res:
        print('\t'.join((str(item[key]) if isinstance(item[key], int) else f"{item[key]:.6f}") for key in keys))
    print('=' * 20)

    metrics = metric_p.findall(content)
    metric_res = [(str(int(step * float(ratio))), value) for value, ratio in metrics]
    print('step\teval_stsb_spearman')
    for item in metric_res:
        print('\t'.join(str(i) for i in item))

@dataclass
class CheckEmbedAndCalSimArguments:
    check_embedding: bool = field(
        default=False,
        metadata={'help': 'Whether to check embedding.'}
    )
    
    model_type: str = field(
        default=None, 
        metadata={
            'help': 'Model type for encoder.',
            'choices': ['bert', 'roberta']
        }
    )

    model_path: str = field(
        default=None, 
        metadata={'help': 'Model parameters path for encoder.'}
    )

    factor: float = field(
        default=0.97,
        metadata={
            'help': 'Dropout decay factor.'
        }
    )

    level_num: int = field(
        default=3,
        metadata={'help': 'The number of embedding hierarchy'}
    )

    dropout_change_layers: int = field(
        default=12,
        metadata={'help': 'The number of layers whose dropout will be changed.'}
    )

    cal_similarity: bool = field(
        default=False,
        metadata={'help': 'Whether to calculate similarities over STSB-test.'}
    )

def check_embedding(args: CheckEmbedAndCalSimArguments):
    if not args.check_embedding:
        return

    # Load transformers' model checkpoint
    model_class: Union[BertForHyper, RobertaForHyper] = \
        BertForHyper if args.model_type == 'bert' else RobertaForHyper
    model = model_class.from_pretrained(args.model_path, dropout_change_layers=args.dropout_change_layers)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    # data
    with open(r'SentEval/data/downstream/STS/STSBenchmark/sts-dev.csv', 'r', encoding='utf8') as fi:
        lines = [line.strip().split('\t') for line in fi.readlines()]
    
    sentences = set()
    for i, line in enumerate(lines):
        if len(line) != 7:
            print(f'Parse Error in line {i}: {line[5:]}')
            continue
        if isinstance(line[5], str):
            sentences.add(line[5])
        if isinstance(line[6], str):
            sentences.add(line[6])
    sentences = list(sentences)

    features = []
    for i in tqdm(range(int(np.ceil(len(sentences) / 128)))):
        sub_sentences = sentences[128 * i: 128 * (i + 1)]
        
        batch = tokenizer(sub_sentences, return_tensors='pt', padding=True,)
    # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)
        sub_features = []
        with torch.no_grad():
            for level in range(args.level_num):
                model._change_dropout(model.bert if hasattr(model, 'bert') else model.roberta,
                                    (1 - 0.9 * args.factor**level))
                sub_features.append(model(**batch, sent_emb=True).pooler_output)
        features.append(torch.stack(sub_features, dim=1)) 
    features = torch.cat(features, dim=0)

    print(Manifold.dist0(features, dim=-1).mean(dim=0).tolist())

def cal_similarity(args: CheckEmbedAndCalSimArguments):
    if not args.cal_similarity:
        return

    # Load transformers' model checkpoint
    model_class: Union[BertForHyper, RobertaForHyper] = \
        BertForHyper if args.model_type == 'bert' else RobertaForHyper
    model = model_class.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # model.custom_param_init((BertHyperConfig if args.model_type == 'bert' 
    #                          else RobertaHyperConfig)()) # test init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # data
    with open(r'SentEval/data/downstream/STS/STSBenchmark/sts-test.csv', 'r', encoding='utf8') as fi:
        lines = [line.strip().split('\t') for line in fi.readlines()]
    
    sentence_pairs = {f'{i}-{i+1}': [] for i in range(5)}

    for i, line in enumerate(lines):
        if len(line) != 7:
            print(f'Parse Error in line {i}: {line[5:]}')
            continue
        score = float(line[4])
        if score < 0 or score > 5:
            print(f'Parse Error in line {i}: wrong similarity score={line[4]}')
            continue
        sent1, sent2 = line[5:]
        if not isinstance(sent1, str) or not isinstance(sent2, str):
            print(f'Parse Error in line {i}: unrecognized sentences={line[5:]}')
            continue
        
        if score == 0:
            sentence_pairs['0-1'].append((sent1, sent2))
        else:
            ceil_score = np.ceil(score).astype(int)
            sentence_pairs[f'{ceil_score - 1}-{ceil_score}'].append((sent1, sent2))

    similarities = {f'{i}-{i+1}': {'cosine': [], 'poincare': []} for i in range(5)}
    min_dist = float('inf')
    for category, pairs in sentence_pairs.items():
        features = []
        for i in tqdm(range(np.ceil(len(pairs) // 64).astype(int))):
            sub_pairs = pairs[i * 64: (i + 1) * 64]
            sub_sentences = []
            for sub_pair in sub_pairs:
                sub_sentences.append(sub_pair[0])
                sub_sentences.append(sub_pair[1])
            batch = tokenizer(sub_sentences, return_tensors='pt', padding=True,)
            for k in batch:
                batch[k] = batch[k].to(device)
            with torch.no_grad():
                feature = model(**batch, sent_emb=True).pooler_output
            features.append(feature.view(-1, 2, feature.shape[-1]))
        features = torch.cat(features, dim=0)
        similarities[category]['cosine'] = torch.cosine_similarity(features[:, 0, :], features[:, 1, :], dim=-1).tolist()
        similarities[category]['poincare'] = (-Manifold.dist(features[:, 0, :], features[:, 1, :], dim=-1)).tolist()
        min_dist = min(min_dist, min(similarities[category]['poincare']))
    
    for category in similarities:
        similarities[category]['poincare'] = [(dist - min_dist/2) / (-min_dist/2)  for dist in similarities[category]['poincare']]
    
    os.makedirs(r'results/cal_similarities', exist_ok=True)
    with open(os.path.join(r'results/cal_similarities', 
                           f'{args.model_type}.{os.path.split(args.model_path)[-1]}.json'),
              'w', encoding='utf8') as fo:
        json.dump(similarities, fo, indent=4)