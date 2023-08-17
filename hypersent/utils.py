import os
import re
import random
# import logging
import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
from transformers import AutoTokenizer, MODEL_FOR_MASKED_LM_MAPPING, TrainingArguments
from transformers.utils import logging
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from hypersent.models import BertForHyper, RobertaForHyper, BertHyperConfig, RobertaHyperConfig
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, List, Optional, Dict, Tuple
from geoopt import PoincareBallExact
from dataclasses import dataclass, field
from enum import Enum

# logger = logging.getLogger(__name__)
logger = logging.get_logger()
Manifold = PoincareBallExact()

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# tools args
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

# train args
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_type: Optional[str] = field(
        # default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES) +\
        ". Also used to determine backend model."
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # HyperSent's arguments
    temp: float = field(
        default=None,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default=None,
        metadata={
            "help": "What kind of pooler to use.",
            "choices": ["cls", "avg", "avg.with_special_tokens", "mask"]
        }
    )
    disable_hyper: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable hyper mode."
        }
    )
    hyperbolic_size: int = field(
        default=None,
        metadata={
            "help": "Embedding size for hyperbolic space."
        }
    )
    num_layers: int = field(
        default=None,
        metadata={
            "help": "Number of training layers in Bert/Roberta."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # HyperSent's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    validation_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The validation data file (.txt or .csv)."}
    )
    only_aigen: bool = field(
        default=False,
        metadata={
            "help": "Whether to train with `aigen` data only. "
        },
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt", "jsonl"], "`train_file` should be a csv, a json(l) or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Training
    hierarchy_type: str = field(
        default="dropout",
        metadata={
            "help": "Hierarchy type for training embeddings in hyperbolic space.",
            "choices": ["dropout", "cluster", "token_cutoff", "aigen"]
        }
    )

    hierarchy_levels: int = field(
        default=4,
        metadata={"help": "Number of Hierarcht level for training embeddings in hyperbolic space."}
    )

    dump_embeddings_num: int = field(
        default=1000,
        metadata={"help": "Number of Embeddings to be dumped after training"}
    )

    dropout_change_layers: int = field(
        default=12,
        metadata={"help": "Number of Layers whose dropout will be changed."}
    )

    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    # overload
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

# Prepare features
@dataclass
class PrepareFeatures:

    column_names: List[str]
    tokenizer: PreTrainedTokenizerBase
    data_args: DataTrainingArguments
    training_args: OurTrainingArguments
    
    def __call__(self, examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.

        total = len(examples[self.column_names[0]])

        if len(self.column_names) == 1: # Unsupervised
            sent_name = self.column_names[0]
            
            # Avoid "None" fields 
            for idx in range(total):
                if examples[sent_name][idx] is None:
                    examples[sent_name][idx] = " "

            sentences = examples[sent_name]
            sent_features = self.tokenizer(
                sentences,
                max_length=self.data_args.max_seq_length,
                truncation=True,
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )

            if self.training_args.hierarchy_type == "cluster" or self.training_args.hierarchy_type == "dropout":

                features = {
                    key: [[value] * (2 if self.training_args.hierarchy_type == "cluster" 
                                            else self.training_args.hierarchy_levels) 
                            for value in values] 
                    for key, values in sent_features.items()
                }

            elif self.training_args.hierarchy_type == "token_cutoff":

                features_keys = list(sent_features) + ['loss_pair']
                features = {key: [] for key in features_keys}
                
                if self.training_args.hierarchy_levels > 10:
                    logger.warn("hierarchy_levels for token_cutoff is limited to no more than 10.")
                hierarchy_levels = min(self.training_args.hierarchy_levels, 10)

                for sent_i in range(len(sentences)):
                    sent_len = len(sent_features[features_keys[0]][sent_i]) - 2 # minus 2 for [CLS] and [SEP]
                    # when sentence is too short, restrict hierarchy_levels
                    available_hl = min(hierarchy_levels, np.ceil(max(0, sent_len - 10) / 4).astype(int) + 1)

                    remove_i = [sent_j + 1 for sent_j in range(sent_len)] # add 1 for [CLS]
                    random.shuffle(remove_i)
                    remove_is = []
                    for remove_token_num in range(hierarchy_levels): # level1 do not remove token
                        if remove_token_num >= available_hl: # hierarchy larger than available_hl is fixed to available_hl
                            remove_is.append(remove_is[-1])
                        else:
                            remove_is.append(remove_i[:remove_token_num])
                    if available_hl > 1: # duplicate for level2
                        remove_is.insert(1, [remove_i[1]])
                    else:
                        remove_is.insert(1, [])

                    for key in features_keys:
                        if key == 'loss_pair':
                            features[key].append(self._get_loss_pair(sent_len, available_hl))
                        else:
                            features[key].append([[feature_c for feature_i, feature_c in enumerate(sent_features[key][sent_i]) 
                                                                    if feature_i not in r_is] 
                                                        for r_is in remove_is])
                    ''' # FIXME
                    for input_ids in features['input_ids'][-1]:
                        print(self.tokenizer.decode(input_ids))
                    print(features['loss_pair'][-1])
                    pdb.set_trace()
                    '''
        
        elif len(self.column_names) == 8: # AIGEN
            
            sentences = []
            if not hasattr(self, 'aigen_keys'): # check which key to use
                self.aigen_keys = []
            if not hasattr(self, 'other_keys'):
                self.other_keys = []

            for idx in range(total):
                if examples['split'][idx] == 'aigen':
                    if len(self.aigen_keys) == 0: # check which key to use
                        for key in ['sentence', '5', '4', '3', '2', '1', '0']:
                            if examples[key][idx] != '':
                                self.aigen_keys.append(key)
                    for key in self.aigen_keys:
                        sentences.append(examples[key][idx] if examples[key][idx] is not None else ' ')
            assert len(sentences) == 0 or len(sentences) % len(self.aigen_keys) == 0
            aigen_group_num = (0 if len(sentences) == 0 else len(sentences) // len(self.aigen_keys))
            
            for idx in range(total):
                if examples['split'][idx] == 'other':
                    if len(self.other_keys) == 0:
                        for key in ['sentence', '5', '4', '3', '2', '1', '0']:
                            if examples[key][idx] != '':
                                self.other_keys.append(key)
                    for key in self.other_keys:
                        sentences.append(examples[key][idx] if examples[key][idx] is not None else ' ')
            assert len(sentences) - aigen_group_num * len(self.aigen_keys) == 0 or \
             (len(sentences) - aigen_group_num * len(self.aigen_keys)) % len(self.other_keys) == 0
            other_group_num = 0 if len(sentences) - aigen_group_num * len(self.aigen_keys) == 0 else\
             (len(sentences) - aigen_group_num * len(self.aigen_keys)) // len(self.other_keys)

            sent_features = self.tokenizer(
                sentences,
                max_length=self.data_args.max_seq_length,
                truncation=True,
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )

            features = {key: [] for key in sent_features.keys()}
            features['split'] = []
            for idx in range(aigen_group_num):
                for key in features.keys():
                    if key == 'split':
                        features[key].append('aigen')
                    else:
                        features[key].append(sent_features[key][idx * len(self.aigen_keys): (idx + 1) * len(self.aigen_keys)])
            for idx in range(other_group_num):
                for key in features.keys():
                    if key == 'split':
                        features[key].append('other')
                    else:
                        features[key].append(sent_features[key][
                            aigen_group_num * len(self.aigen_keys) + idx * len(self.other_keys): 
                            aigen_group_num * len(self.aigen_keys) + (idx + 1) * len(self.other_keys)
                        ])
                        if len(self.other_keys) == 1:
                            features[key][-1] = features[key][-1] * 2

        else:
            raise NotImplementedError
        
        return features

    def _get_loss_pair(self, sent_len: int, available_hl: int) \
            -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:

        distances = []
        for sent_i in range(available_hl):
            for sent_j in range(sent_i + 1, available_hl): # sent_i & sent_j all equals to remove_token_num
                distances.append(((sent_i if sent_i == 0 else sent_i + 1), sent_j + 1, 
                                  (sent_j - sent_i) / (sent_len - sent_j)))
                if sent_i == 1:
                    distances.append((1, sent_j + 1, 
                                  (sent_j - sent_i) / (sent_len - sent_j)))
                elif sent_j == 1: # which means sent_i = 0
                    distances.append((0, 1, 
                                  1 / (sent_len - 1)))
        if available_hl > 1:
            distances.append((1, 2, 2 / (sent_len - 1)))

        distances_pair = []
        for dis_i in range(len(distances)):
            for dis_j in range(dis_i + 1, len(distances)):
                if distances[dis_i][2] < distances[dis_j][2]:
                    distances_pair.append((distances[dis_i][0], distances[dis_i][1],
                                           distances[dis_j][0], distances[dis_j][1]))
                elif distances[dis_i][2] > distances[dis_j][2]:
                    distances_pair.append((distances[dis_j][0], distances[dis_j][1],
                                           distances[dis_i][0], distances[dis_i][1]))
        
        return distances_pair

   
# Data collator
@dataclass
class OurDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    aigen: bool = False
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    # mlm: bool = True
    # mlm_probability: float = data_args.mlm_probability
    aigen_sent_num: int = 0
    other_sent_num: int = 0
    aigen_batch_size: int = 64
    combine_training: bool = False
    aigen_features_cache = []
    other_features_cache = []

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        drop_keys = ['loss_pair', 'split']

        if self.aigen:
            batch_size = len(features)
            for feature in features:
                if feature['split'] == 'aigen':
                    self.aigen_features_cache.append(feature)
                else:
                    self.other_features_cache.append(feature)

            # new
            # if len(self.aigen_features_cache) >= self.aigen_batch_size:
            #     features = self.aigen_features_cache[:self.aigen_batch_size]
            #     self.aigen_features_cache = self.aigen_features_cache[self.aigen_batch_size: ]
            #     if self.combine_training and self.aigen_batch_size < batch_size:
            #         other_batch_size = batch_size - self.aigen_batch_size
            #         features += self.other_features_cache[:other_batch_size]
            #         self.other_features_cache = self.other_features_cache[other_batch_size: ]
            # else:
            #     if self.combine_training:
            #         features = self.other_features_cache[:batch_size]
            #         self.other_features_cache = self.other_features_cache[batch_size:]
            #     else:
            #         features = self.other_features_cache
            #         self.other_features_cache = []

            # old: when testing old version, change to this
            if len(self.aigen_features_cache) >= self.aigen_batch_size:
                features = self.aigen_features_cache
                self.aigen_features_cache = []
                if self.combine_training:
                    features += self.other_features_cache
                    self.other_features_cache = []
            else:
                features = self.other_features_cache
                self.other_features_cache = []

            flat_features = []
            aigen_num = 0
            other_num = 0
            for feature in features:
                if feature['split'] == 'aigen':
                    aigen_num += 1
                    for i in range(self.aigen_sent_num):
                        flat_features.append({k: (v[i] if k in special_keys else v) 
                                             for k, v in feature.items() if k not in drop_keys})
            for feature in features:                             
                if feature['split'] == 'other':
                    other_num += 1
                    for i in range(self.other_sent_num):
                        flat_features.append({k: (v[i] if k in special_keys else v) 
                                             for k, v in feature.items() if k not in drop_keys})
            
            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            # input_ids, attention_mask, token_type_ids：[Tensor of shape(abs, 7, seq_len), Tensor of shape(obs, 2, seq_len)]
            batch = {k : [v[: aigen_num * self.aigen_sent_num], v[aigen_num * self.aigen_sent_num: ]] for k, v in batch.items()}
            for i, sen_num in zip(range(2), [self.aigen_sent_num, self.other_sent_num]):
                for k, v in batch.items():
                    if k in special_keys:
                        v[i] = v[i].reshape((v[i].shape[0] // sen_num if sen_num else 0), sen_num, v[i].shape[1])
                    else:
                        v[i] = v[i].reshape((v[i].shape[0] // sen_num if sen_num else 0), sen_num, v[i].shape[1])[:, 0]
        else:
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: (feature[k][i] if k in special_keys else feature[k]) 
                                        for k in feature if k not in drop_keys})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            # if model_args.do_mlm:
            #     batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            for key in drop_keys:
                if key in features[0]:
                    batch[key] = [feature[key] for feature in features]

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        
        return batch
    
    '''
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    '''

def print_example(example):
    for key in ['sentence'] + [str(i) for i in range(5, 0, -1)]:
        if example[key]:
            print(example[key] + '\n')