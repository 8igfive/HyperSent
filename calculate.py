import re
import os
import pdb
import json
import random
import torch
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from hypersent.models import BertForHyper

DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained('/home/LAB/limx/download/model/bert-base-uncased')

def load_model(model_path: str):
    
    # Load transformers' model checkpoint
    model = BertForHyper.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train()
    return model, tokenizer, device

def load_wiki(txt_path: str, mode: str = 'positive', whn: bool = True):
    with open(txt_path, 'r', encoding='utf8') as fi:
        lines = [line.strip() for line in fi]
    
    if mode == 'positive':
        return [(line, line) for line in lines]
    elif mode == 'negative':
        another_lines = lines.copy()
        random.shuffle(another_lines)
        return [(line0, line1) for line0, line1 in zip(lines, another_lines)]
    elif mode == 'all':
        return lines
    else:
         raise NotImplementedError

def load_nli(csv_path: str, mode: str = 'positive', whn: bool = True):
    df = pd.read_csv(csv_path)
    lines = []
    for row in df.iloc:
        lines.append((row['sent0'], row['sent1'], row['hard_neg']))
    
    if mode == 'positive':
        return [(line[0], line[1]) for line in lines]
    elif mode == 'negative':
        if whn:
            return [(line[0], line[2]) for line in lines]
        else:
            another_lines = lines.copy()
            random.shuffle(another_lines)
            return [(line0[0], line1[0]) for line0, line1 in zip(lines, another_lines)]
    elif mode == 'all':
        res = []
        for line in lines:
            res.append(line[0])
            res.append(line[1])
            if whn:
                res.append(line[2])
        return res
    else:
        raise NotImplementedError

def load_stsb(split: str, mode: str = 'positive', whn: bool = True):
    stsb = load_dataset(r'mteb/stsbenchmark-sts', split=split)

    if mode == 'positive':
        return [(row['sentence1'], row['sentence2']) for row in stsb if row['score'] >= 4.]
    elif mode == 'negative':
        return [(row['sentence1'], row['sentence2']) for row in stsb if row['score'] <= 1.]
    elif mode == 'all':
        res = []
        for row in stsb:
            res.append(row['sentence1'])
            res.append(row['sentence2'])
        return res
    else:
        raise NotImplementedError

def load_aigen(path: str, mode: str = 'positive', whn: bool = True, multiple_positive: bool = False):
    with open(path, 'r', encoding='utf8') as fi:
        rows = [json.loads(row) for row in fi]
        aigens = [tuple(row[key] for key in ['sentence', '5', '4', '3', '2', '1', '0'] if row[key] != '') 
                  for row in rows if row['split'] == 'aigen']
        others = [row['sentence'] for row in rows if row['split'] == 'other']
    
    if mode == 'positive':
        # return [(row[0], row[1]) for row in aigens] + [(row, row) for row in others]
        if multiple_positive:
            res = []
            for row in aigens:
                for i in range(1, len(row) - 1):
                    res.append((row[0], row[i]))
            return res
        else:
            return [(row[0], row[1]) for row in aigens]
    elif mode == 'negative':
        if whn:
            another_others = others.copy()
            random.shuffle(another_others)
            # return [(row[0], row[-1]) for row in aigens] + [(row0, row1) for row0, row1 in zip(others, another_others)]
            return [(row[0], row[-1]) for row in aigens]
        else:
            alls = aigens + others
            another_alls = alls.copy()
            random.shuffle(another_alls)
            # return [(row0[0] if isinstance(row0, tuple) else row0, 
            #          row1[0] if isinstance(row1, tuple) else row1) for row0, row1 in zip(alls, another_alls)]
            
            another_aigens = aigens.copy()
            random.shuffle(another_aigens)
            return [(row0[0], row1[0]) for row0, row1 in zip(aigens, another_aigens)]
        # return [(row[0], row[-1]) for row in aigens]
    elif mode == 'all':
        res = []
        for row in aigens:
            for sentence in (row if whn else row[:-1]):
                res.append(sentence)
        # for row in others:
        #     res.append(row)
        return res
    else:
        raise NotImplementedError

def load_other_augmentation(path: str, mode: str = 'positive', whn: bool = True):
    with open(path, 'r', encoding='utf8') as fi:
        rows = [json.loads(row) for row in fi]
        aigens = [(row['sentence'], row['5'])
                  for row in rows]
    
    if mode == 'positive':
        return [(row[0], row[1]) for row in aigens]
    elif mode == 'negative':
        sent0 = [row[0] for row in aigens]
        sent1 = [row[1] for row in aigens]
        random.shuffle(sent1)
        return [(s0, s1) for s0, s1 in zip(sent0, sent1)]
    elif mode == 'all':
        res = []
        for row in aigens:
            res.append(row[0])
            res.append(row[1])
        return res
    else:
        raise NotImplementedError

def batcher(sentences, model, tokenizer, device, max_length=512):
    # Tokenization
    if max_length is not None:
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=True
        )
    else:
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )

    # Move to the correct device
    for k in batch:
        batch[k] = batch[k].to(device)
    
    # Get raw embeddings
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

    return pooler_output.cpu()

def cal_embeddings(model, tokenizer, device, sentences, batch_size=256):
    paired = isinstance(sentences[0], tuple)

    b_sentences = []
    features = []
    for sentence in tqdm(sentences):
        if paired:
            for l_sentence in sentence:
                b_sentences.append(l_sentence)
        else:
            b_sentences.append(sentence)

        if len(b_sentences) >= batch_size:
            features.append(batcher(b_sentences, model, tokenizer, device))
            b_sentences = []
    if len(b_sentences) > 0:
        features.append(batcher(b_sentences, model, tokenizer, device))
        b_sentences = []
    
    if paired:
        for i, b_features in enumerate(features):
            features[i] = b_features.reshape(-1, len(sentences[0]), b_features.shape[-1])
    
    features = torch.cat(features, dim=0)
    return features

def collect_embeddings(ckpt_dir, sentences, max_size=None, 
    base_model: str = '/home/LAB/limx/download/model/bert-base-uncased'):
    save_name = 'ckpts_features_paired' if isinstance(sentences[0], tuple) else 'ckpts_features'
    if max_size is not None:
        save_name = f'{save_name}_{max_size}.pt'
        random.shuffle(sentences)
        sentences = sentences[:max_size]
    else:
        save_name = f'{save_name}.pt'

    if save_name in os.listdir(ckpt_dir):
        print('Cached ckpt_features found, loading.')
        ckpts_features = torch.load(os.path.join(ckpt_dir, save_name))
    else:
        ckpts = [os.path.join(ckpt_dir, ckpt) for ckpt in os.listdir(ckpt_dir) if 'checkpoint' in ckpt]
        
        ckpts_features = {
            0: cal_embeddings(*load_model(base_model), sentences)
        }
        for ckpt in tqdm(ckpts):
            step = int(os.path.split(ckpt)[-1][len('checkpoint-'): ])
            model, tokenizer, decive = load_model(ckpt)
            ckpts_features[step] = cal_embeddings(model, tokenizer, decive, sentences)
        
        torch.save(ckpts_features, os.path.join(ckpt_dir, save_name))

    return ckpts_features

def align_loss(x, y, alpha=2):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def cal_alignment_and_uniformity(ckpt_dir, corpus_path, whn, load_fn, 
    cal_aligment=True, cal_uniformity=True, 
    alignment_max_size=None, uniformity_max_size=None):

    # alignment
    if cal_aligment:
        print('Calculate Alignment:\n')
        sentences = load_fn(corpus_path, mode='positive', whn=whn)
        ckpts_features = collect_embeddings(ckpt_dir, sentences, alignment_max_size)
        ckpts_alignment = {}
        for step, features in tqdm(ckpts_features.items()):
            ckpts_alignment[step] = align_loss(features[:, 0], features[:, 1])

        alignments = sorted(((step, alignment) for step, alignment in ckpts_alignment.items()), key=lambda x: x[0])
        with open(os.path.join(ckpt_dir, 'alignments'), 'w', encoding='utf8') as fo:
            fo.write('\n'.join(f'{step}\t\t{alignment}' for step, alignment in alignments))
        for step, alignment in alignments:
            print(f'{step}\t\t{alignment}')

    # uniformity
    if cal_uniformity:
        print('Calculate Uniformity:\n')
        sentences = load_fn(corpus_path, mode='all', whn=whn)
        ckpts_features = collect_embeddings(ckpt_dir, sentences, uniformity_max_size)
        ckpts_uniformity = {}
        for step, features in tqdm(ckpts_features.items()):
            if uniformity_max_size:
                features = features[: uniformity_max_size]
            ckpts_uniformity[step] = uniform_loss(features)
        
        uniformities = sorted(((step, uniformity) for step, uniformity in ckpts_uniformity.items()), key=lambda x: x[0])
        with open(os.path.join(ckpt_dir, 'uniformities'), 'w', encoding='utf8') as fo:
            fo.write('\n'.join(f'{step}\t\t{uniformity}' for step, uniformity in uniformities))
        for step, uniformity in uniformities:
            print(f'{step}\t\t{uniformity}')

def cal_anu_bbu(base_model, corpus_path, load_fn, 
    cal_aligment=True, cal_uniformity=True, 
    alignment_max_size=None, uniformity_max_size=None):
    # alignment
    if cal_aligment:
        print('Calculate Alignment:\n')
        sentences = load_fn(corpus_path, mode='positive')
        if alignment_max_size:
            sentences = sentences[:alignment_max_size]
        features = cal_embeddings(*load_model(base_model), sentences)
        alignment = align_loss(features[:, 0], features[:, 1])
        print(f'0\t\t{alignment}')

    # uniformity
    if cal_uniformity:
        print('Calculate Uniformity:\n')
        sentences = load_fn(corpus_path, mode='all')
        if uniformity_max_size:
            sentences = sentences[:uniformity_max_size]
        features = cal_embeddings(*load_model(base_model), sentences)
        uniformity = uniform_loss(features)
        print(f'0\t\t{uniformity}')
    
    return alignment, uniformity

def assemble_alignment_and_uniformity(ckpt_dir):

    with open(os.path.join(ckpt_dir, 'alignments'), 'r', encoding='utf8') as fi:
        alignments = sorted(((int(step), float(alignment)) for step, alignment in (row.strip().split() for row in fi)), key=lambda x: x[0])

    with open(os.path.join(ckpt_dir, 'uniformities'), 'r', encoding='utf8') as fi:
        uniformities = sorted(((int(step), float(uniformity)) for step, uniformity in (row.strip().split() for row in fi)), key=lambda x: x[0])
    
    anu = {'step':[], 'alignment': [], 'uniformity': []}
    for sa, su in zip(alignments, uniformities):
         assert sa[0] == su[0]
         anu['step'].append(sa[0])
         anu['alignment'].append(sa[1])
         anu['uniformity'].append(su[1])
    df = pd.DataFrame(anu)
    df.to_csv(os.path.join(ckpt_dir, 'alignment_and_uniformity.csv'), index=False)

def collect_loss_and_eval(log_path: str):
    with open(log_path, 'r', encoding='utf8') as fi:
        content = fi.read()
    losses = re.findall(r"'loss': ([\d\.]+),?", content)
    stsbs = re.findall(r"'eval_stsb_spearman': ([\d\.]+),", content)
    sickrs = re.findall(r"'eval_sickr_spearman': ([\d\.]+),", content)
    avgs = re.findall(r"'eval_avg_sts': ([\d\.]+),", content)
    data = {
        'step': [125 * i for i in range(len(losses))], # FIXME: check this
        'loss': [loss for loss in losses],
        'stsb': [stsb for stsb in stsbs],
        'sickr': [sickr for sickr in sickrs],
        'avg': [avg for avg in avgs]
    }

    df = pd.DataFrame(data)
    path_names = list(os.path.split(log_path))
    path_names[-1] = f"{path_names[-1].split('.')[0]}.csv"
    df.to_csv(os.path.join(*path_names), index=False)

def cal_mers(sentence_pairs, 
    tokenizer = lambda sentences: DEFAULT_TOKENIZER.batch_encode_plus(sentences, add_special_tokens=False)['input_ids']):
    def _cal_mers(ids_pair):
        ed_cache = [[j if i == 0 else (i if j == 0 else 0) 
                     for j in range(len(ids_pair[1]) + 1)] 
                    for i in range(len(ids_pair[0]) + 1)]
        # 1 for left&up, 2 for up, 3 for left 
        dir_cache = [[3 if i == 0 else (2 if j == 0 else 0) 
                     for j in range(len(ids_pair[1]) + 1)] 
                    for i in range(len(ids_pair[0]) + 1)]
        dir_cache[0][0] = 0
        for i in range(1, len(ids_pair[0]) + 1):
            for j in range(1, len(ids_pair[1]) + 1):
                if ids_pair[0][i - 1] == ids_pair[1][j - 1]:
                    ed_cache[i][j] = ed_cache[i-1][j-1]
                    dir_cache[i][j] = 1
                else:
                    if ed_cache[i - 1][j - 1] <= ed_cache[i - 1][j]:
                        tmp_min_ed = ed_cache[i - 1][j - 1]
                        dir_cache[i][j] = 1
                    else:
                        tmp_min_ed = ed_cache[i - 1][j]
                        dir_cache[i][j] = 2
                    if ed_cache[i][j - 1] < tmp_min_ed:
                        tmp_min_ed = ed_cache[i][j - 1]
                        dir_cache[i][j] = 3
                    ed_cache[i][j] = tmp_min_ed + 1
        
        i = len(ids_pair[0])
        j = len(ids_pair[1])
        ed = ed_cache[i][j]
        td = 0
        while dir_cache[i][j] != 0:
            td += 1
            if dir_cache[i][j] == 1:
                i -= 1
                j -= 1
            elif dir_cache[i][j] == 2:
                i -= 1
            else: # 3
                j -= 1
        if td == 0:
            return 0
        return ed / td
    mers = []
    for sentence_pair in tqdm(sentence_pairs):
        ids_pair = tokenizer(list(sentence_pair))
        mers.append(_cal_mers(ids_pair))
    
    return mers

def cal_ious(sentence_pairs, 
    tokenizer = lambda sentences: DEFAULT_TOKENIZER.batch_encode_plus(sentences, add_special_tokens=False)['input_ids']):
    ious = []
    for sentence_pair in tqdm(sentence_pairs):
        ids_pair = tokenizer(list(sentence_pair))
        idx_set0 = set(ids_pair[0])
        idx_set1 = set(ids_pair[1])
        if len(idx_set1) == 0:
            ious.append(0)
        else:
            ious.append(len(idx_set0 & idx_set1) / len(idx_set0 | idx_set1))
    return ious

def cal_cosine(sentence_pairs, model_path='/home/LAB/limx/download/model/bert-base-uncased', max_sentence_num=10000):
    model, tokenizer, device = load_model(model_path)
    if max_sentence_num > 0:
        random.shuffle(sentence_pairs)
        sentence_pairs = sentence_pairs[:max_sentence_num]

    features = cal_embeddings(model, tokenizer, device, sentence_pairs)
    cosine_similarities = F.cosine_similarity(features[:, 0], features[:, 1], dim=-1)
    return cosine_similarities.tolist()

def collect_metrics4corpus(load_fn, whn, corpus_arg, mode='positive', metrics='mer', dump_dir = None, max_num = None):
    sentence_pairs = load_fn(corpus_arg, mode=mode, whn=whn)
    if max_num:
        random.shuffle(sentence_pairs)
        sentence_pairs = sentence_pairs[:max_num]
    if not dump_dir:
        dump_dir = os.path.join(r'result/metrics', load_fn.__name__.split('_')[-1])
    os.makedirs(dump_dir, exist_ok=True)

    if metrics == 'mer':
        ms = cal_mers(sentence_pairs)
    elif metrics == 'iou':
        ms = cal_ious(sentence_pairs)
    elif metrics == 'cosine':
        ms = cal_cosine(sentence_pairs)
    else:
        raise NotImplementedError
    
    with open(os.path.join(dump_dir, f'{mode}_{metrics}'), 'w', encoding='utf8') as fo:
        fo.write('\n'.join(str(m) for m in ms))

def collect_intra_adj_num(model, tokenizer, device, sentences, cache_dir, instance_name, dump_dir, 
    max_num=10000, adj_max_distance = 4., split='all'):
    if isinstance(sentences[0], tuple):
        n_sentences = []
        for sentence_pair in sentences:
            for sentence in sentence_pair:
                n_sentences.append(sentence)
        sentences = n_sentences
    
    sts_embeddings_path = os.path.join(cache_dir, 'sts_embeddings_path.pt')
    if os.path.exists(sts_embeddings_path):
        print('Cache found, loading sts_embeddings... ')
        sts_embeddings = torch.load(sts_embeddings_path)
    else:
        sts_sentences = load_stsb('validation', mode='all')
        sts_embeddings = cal_embeddings(model, tokenizer, device, sts_sentences)
        torch.save(sts_embeddings, sts_embeddings_path)

    base_dir = os.path.join(cache_dir, instance_name)
    os.makedirs(base_dir, exist_ok=True)
    distances_path = os.path.join(base_dir, 'distances.pt') if split == 'all' else os.path.join(base_dir, f'distances_{split}.pt')
    if os.path.exists(distances_path):
        print('Cache found, loading distances... ')
        distances = torch.load(distances_path)
    else:
        embeddings_path = os.path.join(base_dir, 'embeddings.pt') if split == 'all' else os.path.join(base_dir, f'embeddings_{split}.pt')
        if os.path.exists(embeddings_path):
            print('Cache found, loading embeddings... ')
            embeddings = torch.load(embeddings_path)
        else:
            random.shuffle(sentences)
            embeddings = cal_embeddings(model, tokenizer, device, sentences[:max_num])
            torch.save(embeddings, embeddings_path)
        distances = torch.norm(sts_embeddings[:, None] - embeddings[None], dim=-1)
        torch.save(distances, distances_path)
    avg_adj_num = (distances <= adj_max_distance).sum(dim=-1).float().mean()
    
    os.makedirs(dump_dir, exist_ok=True)
    save_path = os.path.join(dump_dir, 'avg_adj_num' if split == 'all' else f'avg_adj_num_{split}')
    with open(save_path, 'w', encoding='utf8') as fo:
        fo.write(f"{avg_adj_num.item()}")
    print(f"{avg_adj_num.item()}")

if __name__ == '__main__':

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # cal_alignment_and_uniformity(r'results/runs/test_token_shuffle', 
    #     r'validation', load_stsb, 
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_token_cutoff', 
    #     r'validation', load_stsb,
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_nli_whn', 
    #     r'validation', load_stsb, 
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_nli_wohn', 
    #     r'validation', load_stsb, 
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_sts_whn', 
    #     r'validation', load_stsb, 
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_sts_wohn', 
    #     r'validation', load_stsb, 
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_ori_nli_whn', 
    #     r'validation', load_stsb, 
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_ori_nli_wohn', 
    #     r'validation', load_stsb,  
    #     alignment_max_size=10000, uniformity_max_size=10000)
    
    # cal_alignment_and_uniformity(r'results/runs/test_align_nli_whn',
    #     r'validation', load_stsb,  
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_align_nli_wohn',
    #     r'validation', load_stsb,  
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_align_sts_whn',
    #     r'validation', load_stsb,  
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_align_sts_wohn',
    #     r'validation', load_stsb,
    #     alignment_max_size=10000, uniformity_max_size=10000)
    
    # cal_alignment_and_uniformity(r'results/runs/test_sup_aigen_nli_whn',
    #     r'validation', load_stsb,  
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_sup_aigen_nli_wohn',
    #     r'validation', load_stsb,  
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_sup_aigen_sts_whn',
    #     r'validation', load_stsb,  
    #     alignment_max_size=10000, uniformity_max_size=10000)
    # cal_alignment_and_uniformity(r'results/runs/test_sup_aigen_sts_wohn',
    #     r'validation', load_stsb,  
    #     alignment_max_size=10000, uniformity_max_size=10000)

    # assemble_alignment_and_uniformity(r'results/runs/test_token_shuffle')
    # assemble_alignment_and_uniformity(r'results/runs/test_token_cutoff')
    # assemble_alignment_and_uniformity(r'results/runs/test_nli_whn')
    # assemble_alignment_and_uniformity(r'results/runs/test_nli_wohn')
    # assemble_alignment_and_uniformity(r'results/runs/test_sts_whn')
    # assemble_alignment_and_uniformity(r'results/runs/test_sts_wohn')
    # assemble_alignment_and_uniformity(r'results/runs/test_ori_nli_whn')
    # assemble_alignment_and_uniformity(r'results/runs/test_ori_nli_wohn')
    
    # assemble_alignment_and_uniformity(r'results/runs/test_align_nli_whn')
    # assemble_alignment_and_uniformity(r'results/runs/test_align_nli_wohn')
    # assemble_alignment_and_uniformity(r'results/runs/test_align_sts_whn')
    # assemble_alignment_and_uniformity(r'results/runs/test_align_sts_wohn')

    # assemble_alignment_and_uniformity(r'results/runs/test_sup_aigen_nli_whn')
    # assemble_alignment_and_uniformity(r'results/runs/test_sup_aigen_nli_wohn')
    # assemble_alignment_and_uniformity(r'results/runs/test_sup_aigen_sts_whn')
    # assemble_alignment_and_uniformity(r'results/runs/test_sup_aigen_sts_wohn')

    # max_num=1000
    # for split in tqdm(['train', 'eval']):
    #     for ratio in tqdm(range(10, 11)):
    #         ckpt_path = os.path.join(r'results/runs/ratios', f'test_nli_wiki_{ratio}e-1')
    #         data_path = os.path.join(r'data/230804/ratios', f'nli_wiki_{ratio}e-1_heldout.jsonl') if split == 'train' else 'validation'
    #         load_fn = load_aigen if split == 'train' else load_stsb
    #         cal_alignment_and_uniformity(ckpt_path, data_path, load_fn, 
    #                                     alignment_max_size=max_num, uniformity_max_size=max_num)
    #         assemble_alignment_and_uniformity(ckpt_path)
    #         dst_dir = os.path.join(ckpt_path, split)
    #         os.makedirs(dst_dir)
    #         for file_name in ['alignment_and_uniformity.csv', 'alignments', 'uniformities',
    #                           f'ckpts_features_{max_num}.pt', f'ckpts_features_paired_{max_num}.pt']:
    #             os.system(f'mv {os.path.join(ckpt_path, file_name)} {dst_dir}')

    
    '''
    max_num = 1000
    # ckpt_paths = [
    #     r'results/runs/aigens/test_wiki_nli_whn', r'results/runs/aigens/test_wiki_nli_wohn',
    #     r'results/runs/aigens/test_wiki_sts_whn', r'results/runs/aigens/test_wiki_sts_wohn',
    #     r'results/runs/aigens/test_nli_nli_whn', r'results/runs/aigens/test_nli_nli_wohn',
    #     r'results/runs/aigens/test_nli_sts_whn', r'results/runs/aigens/test_nli_sts_wohn'
    # ]
    # data_paths = [
    #     r'data/230729/wiki1m_aigen_nli_20k_2_heldout.jsonl', r'data/230729/wiki1m_aigen_nli_20k_2_heldout.jsonl',
    #     r'data/230718/wiki1m_aigen_remove_negative_20k_4_heldout.json', r'data/230718/wiki1m_aigen_remove_negative_20k_4_heldout.json',
    #     r'data/230802/nli20k_aigen_new_nli_2_heldout.jsonl', r'data/230802/nli20k_aigen_new_nli_2_heldout.jsonl',
    #     r'data/230802/nli20k_aigen_sts_3_heldout.jsonl', r'data/230802/nli20k_aigen_sts_3_heldout.jsonl'
    # ]
    ckpt_paths = [
        r'results/runs/aigens/test_nli_nli_whn', r'results/runs/aigens/test_nli_nli_wohn',
        r'results/runs/aigens/test_nli_sts_whn', r'results/runs/aigens/test_nli_sts_wohn'
    ]
    data_paths = [
        r'data/230807/nliunsup_aigen_nli_2_20k_heldout.jsonl', r'data/230807/nliunsup_aigen_nli_2_20k_heldout.jsonl',
        r'data/230807/nliunsup_aigen_sts_3_20k_heldout.jsonl', r'data/230807/nliunsup_aigen_sts_3_20k_heldout.jsonl'
    ]
    for split in tqdm(['eval']):# 'train',
        for ckpt_path, data_path in zip(ckpt_paths, data_paths):
            load_fn = load_aigen
            if split == 'eval':
                data_path = 'validation'
                load_fn = load_stsb
            whn = 'whn' in ckpt_path
            cal_alignment_and_uniformity(ckpt_path, data_path, whn, load_fn, 
                                         alignment_max_size=max_num, uniformity_max_size=max_num)
            assemble_alignment_and_uniformity(ckpt_path)
            dst_dir = os.path.join(ckpt_path, split)
            os.makedirs(dst_dir)
            for file_name in ['alignment_and_uniformity.csv', 'alignments', 'uniformities',
                              f'ckpts_features_{max_num}.pt', f'ckpts_features_paired_{max_num}.pt']:
                os.system(f'mv {os.path.join(ckpt_path, file_name)} {dst_dir}')
    '''

    '''
    for base in ['nli']: # 'wiki', 
        for llm_method in ['nli', 'sts']:
            for setting in ['whn', 'wohn']:
                collect_loss_and_eval(os.path.join(r'results/logs/aigens', f'test_{base}_{llm_method}_{setting}.log'))
    '''

    max_num = 1000
    ckpt_paths = [
        # r'results/runs/aigens/test_sts/test_nli_sts_triplet_10e-2p2wohn',
        r'results/runs/aigens/test_sts/test_wiki_sts_triplet_new_10e-2p2'
    ]
    data_paths = [
        # r'data/230807/nliunsup_aigen_sts_3_20k_heldout.jsonl',
        r'data/230728/wiki1m_aigen_remove_negative_20k_3_heldout.jsonl'
    ]
    for split in tqdm(['train','eval']): 
        for ckpt_path, data_path in zip(ckpt_paths, data_paths):
            load_fn = lambda path, mode, whn : load_aigen(path, mode, whn, True)
            if split == 'eval':
                data_path = 'validation'
                load_fn = load_stsb
            whn = 'whn' in ckpt_path
            cal_alignment_and_uniformity(ckpt_path, data_path, whn, load_fn, 
                                         alignment_max_size=max_num, uniformity_max_size=max_num)
            assemble_alignment_and_uniformity(ckpt_path)
            dst_dir = os.path.join(ckpt_path, split)
            os.makedirs(dst_dir)
            for file_name in ['alignment_and_uniformity.csv', 'alignments', 'uniformities',
                              f'ckpts_features_{max_num}.pt', f'ckpts_features_paired_{max_num}.pt']:
                os.system(f'mv {os.path.join(ckpt_path, file_name)} {dst_dir}')
    collect_loss_and_eval(r'results/logs/aigens/test_sts/test_nli_sts_triplet_10e-2p2wohn.log')
    collect_loss_and_eval(r'results/logs/aigens/test_sts/test_wiki_sts_triplet_new_10e-2p2.log')

    '''
    # ckpt_paths = [r'results/runs/test_token_cutoff']
    # data_paths = [r'data/230730/wiki1m_token_cutoff_0.10_heldout.jsonl']
    # for split in tqdm(['eval']): # 'train', 
    #     for ckpt_path, data_path in zip(ckpt_paths, data_paths):
    #         load_fn = load_other_augmentation
    #         if split == 'eval':
    #             data_path = 'validation'
    #             load_fn = load_stsb
    #         cal_alignment_and_uniformity(ckpt_path, data_path, True, load_fn, 
    #                                      alignment_max_size=max_num, uniformity_max_size=max_num)
    #         assemble_alignment_and_uniformity(ckpt_path)
    #         dst_dir = os.path.join(ckpt_path, split)
    #         os.makedirs(dst_dir)
    #         for file_name in ['alignment_and_uniformity.csv', 'alignments', 'uniformities',
    #                           f'ckpts_features_{max_num}.pt', f'ckpts_features_paired_{max_num}.pt']:
    #             os.system(f'mv {os.path.join(ckpt_path, file_name)} {dst_dir}')
    '''


    # collect_loss_and_eval(r'result/log/unsup.log')
    # collect_loss_and_eval(r'result/log/sup.log')
    # collect_loss_and_eval(r'results/logs/test_token_shuffle.log')
    # collect_loss_and_eval(r'results/logs/test_token_cutoff.log')
    # collect_loss_and_eval(r'results/logs/test_nli_whn.log')
    # collect_loss_and_eval(r'results/logs/test_nli_wohn.log')
    # collect_loss_and_eval(r'results/logs/test_sts_whn.log')
    # collect_loss_and_eval(r'results/logs/test_sts_wohn.log')
    # collect_loss_and_eval(r'results/logs/test_ori_nli_whn.log')
    # collect_loss_and_eval(r'results/logs/test_ori_nli_wohn.log')

    # collect_loss_and_eval(r'results/logs/test_align_nli_whn.log')
    # collect_loss_and_eval(r'results/logs/test_align_nli_wohn.log')
    # collect_loss_and_eval(r'results/logs/test_align_sts_whn.log')
    # collect_loss_and_eval(r'results/logs/test_align_sts_wohn.log')

    # collect_loss_and_eval(r'results/logs/test_sup_aigen_nli_whn.log')
    # collect_loss_and_eval(r'results/logs/test_sup_aigen_nli_wohn.log')
    # collect_loss_and_eval(r'results/logs/test_sup_aigen_sts_whn.log')
    # collect_loss_and_eval(r'results/logs/test_sup_aigen_sts_wohn.log')
    
    # for ratio in range(11):
    #     log_path = f'results/logs/ratios/test_nli_wiki_{ratio}e-1.log'
    #     collect_loss_and_eval(log_path)

    # sentence_pairs = [
    #     ['Bryan Cranston will return as Walter White for breaking bad spin off, report claims.',
    #      'It has been reported that Bryan Cranston will reprise his role as Walter White in a spin-off of Breaking Bad.'],
    #     ['Bryan Cranston will return as Walter White for breaking bad spin off, report claims.',
    #       'Bryan Cranston will not return as Walter White for Breaking Bad spin off, report claims.'],
    #     ['Bryan Cranston will return as Walter White for breaking bad spin off, report claims.',
    #      'Bryan Cranston will return as Walter White for breaking bad spin off, a latest report claims.'],
    #     ['Bryan Cranston will return as Walter White for breaking bad spin off, report claims.',
    #      'Bryan Cranston will come back as Walter White for Breaking Bad spin off, report claims.'],
    #     ['Bryan Cranston will return as Walter White for breaking bad spin off, report claims.',
    #      'Digital era threatens future of driveins.'], 
    # ]
    # print(cal_mer(sentence_pairs))
    # collect_metrics4corpus(load_nli, r'data/nli_train.csv', mode='negative', metrics='mer')
    # collect_metrics4corpus(load_wiki, r'data/wiki1m_train.txt', mode='negative', metrics='mer')
    # collect_metrics4corpus(load_stsb, r'validation', mode='negative', metrics='mer')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230718/wiki1m_aigen_remove_negative_20k_4_train.json', mode='negative', metrics='mer')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230729/wiki1m_aigen_nli_20k_2_train.jsonl', mode='negative', metrics='mer')

    # collect_metrics4corpus(load_nli, r'data/nli_train.csv', mode='negative', metrics='iou')
    # collect_metrics4corpus(load_wiki, r'data/wiki1m_train.txt', mode='negative', metrics='iou')
    # collect_metrics4corpus(load_stsb, r'validation', mode='negative', metrics='iou')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230718/wiki1m_aigen_remove_negative_20k_4_train.json', 
    #                        mode='positive', metrics='mer')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230718/wiki1m_aigen_remove_negative_20k_4_train.json', 
    #                        mode='negative', metrics='mer')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230718/wiki1m_aigen_remove_negative_20k_4_train.json', 
    #                        mode='positive', metrics='iou')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230718/wiki1m_aigen_remove_negative_20k_4_train.json', 
    #                        mode='negative', metrics='iou')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230729/wiki1m_aigen_nli_20k_2_train.jsonl', 
    #                        mode='positive', metrics='mer')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230729/wiki1m_aigen_nli_20k_2_train.jsonl', 
    #                        mode='negative', metrics='mer')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230729/wiki1m_aigen_nli_20k_2_train.jsonl', 
    #                        mode='positive', metrics='iou')
    # collect_metrics4corpus(load_aigen, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230729/wiki1m_aigen_nli_20k_2_train.jsonl', 
    #                        mode='negative', metrics='iou')


    # collect_metrics4corpus(load_other_augmentation, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230730/wiki1m_token_shuffle_train.jsonl',
    #     mode='positive', metrics='mer')
    # collect_metrics4corpus(load_other_augmentation, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230730/wiki1m_token_shuffle_train.jsonl',
    #     mode='positive', metrics='iou')
    # collect_metrics4corpus(load_other_augmentation, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230730/wiki1m_token_shuffle_train.jsonl',
    #     mode='negative', metrics='mer')
    # collect_metrics4corpus(load_other_augmentation, r'/home/LAB/limx/project/hyperbolic/HyperSent/data/230730/wiki1m_token_shuffle_train.jsonl',
    #     mode='negative', metrics='iou')
        
    '''
    # data_paths = [
    #     r'data/230729/wiki1m_aigen_nli_20k_2_train.jsonl',
    #     r'data/230718/wiki1m_aigen_remove_negative_20k_4_train.json',
    #     r'data/230802/nli20k_aigen_new_nli_2_train.jsonl',
    #     r'data/230802/nli20k_aigen_sts_3_train.jsonl'
    # ]
    # data_labels = [
    #     'wiki_nli', 'wiki_sts', 
    #     'nli_nli', 'nli_sts'
    # ]
    data_paths = [
        r'data/230807/nliunsup_aigen_nli_2_20k_train.jsonl',
        r'data/230807/nliunsup_aigen_sts_3_20k_train.jsonl'
    ]
    data_labels = [
        'nli_nli', 'nli_sts'
    ]
    for split in tqdm(['positive', 'negative']):
        for metric in tqdm(['mer', 'iou', 'cosine']):
            for data_path, data_label in tqdm(zip(data_paths, data_labels)):
                for whn in tqdm([True, False]):
                    dump_dir = os.path.join(r'results/metrics/aigens', data_label, 'whn' if whn else 'wohn', 'inter')
                    collect_metrics4corpus(load_aigen, whn, data_path, mode=split, metrics=metric, dump_dir=dump_dir, max_num=10000)
    '''

    # for split in tqdm(['positive', 'negative']):
    #     for metric in tqdm(['mer', 'iou', 'cosine']):
    #         for whn in tqdm([True, False]):
    #             dump_dir = os.path.join(r'results/metrics/nli', 'whn' if whn else 'wohn', 'inter')
    #             collect_metrics4corpus(load_nli, whn, r'/home/LAB/limx/project/hyperbolic/SimCSE/data/nli_train.csv',
    #                                    mode=split, metrics=metric, dump_dir=dump_dir, max_num=10000)

    # cal_anu_bbu('bert-base-uncased', r'/home/LAB/limx/project/hyperbolic/SimCSE/data/wiki1m_heldout.txt',
    #             load_wiki, alignment_max_size=10000, uniformity_max_size=10000)
    # cal_anu_bbu('bert-base-uncased', r'/home/LAB/limx/project/hyperbolic/SimCSE/data/nli_heldout.csv',
    #             load_nli, alignment_max_size=10000, uniformity_max_size=10000)

    '''
    model, tokenizer, device = load_model('/home/LAB/limx/download/model/bert-base-uncased')
    cache_dir = r'results/cache'
    load_fns = [
        load_aigen, load_aigen, load_aigen, load_aigen
    ]
    # data_paths = [
    #     r'data/230729/wiki1m_aigen_nli_20k_2_train.jsonl',
    #     r'data/230718/wiki1m_aigen_remove_negative_20k_4_train.json',
    #     r'data/230802/nli20k_aigen_new_nli_2_train.jsonl',
    #     r'data/230802/nli20k_aigen_sts_3_train.jsonl',
    # ]
    # instance_names = [
    #     'wiki_nli', 'wiki_sts', 'nli_nli', 'nli_sts'
    # ]
    data_paths = [
        r'data/230807/nliunsup_aigen_nli_2_20k_train.jsonl',
        r'data/230807/nliunsup_aigen_sts_3_20k_train.jsonl'
    ]
    instance_names = [
        'nli_nli', 'nli_sts'
    ]
    
    for load_fn, data_path, instance_name in zip(load_fns, data_paths, instance_names):
        for whn in [True, False]:
            instance = f"aigen_{instance_name}_{'whn' if whn else 'wohn'}"
            dump_dir = os.path.join(r'results/metrics/aigens', instance_name, 'whn' if whn else 'wohn', 'intra')
            for split in ['positive', 'negative']:
                sentences = load_fn(data_path, split, whn)
                collect_intra_adj_num(model, tokenizer, device,
                sentences, cache_dir, instance, dump_dir,
                max_num=10000, adj_max_distance=4., split=split)
    '''

    '''
    for whn in [True, False]:
        instance = f"nli_{'whn' if whn else 'wohn'}"
        dump_dir = os.path.join(r'results/metrics/nli', 'whn' if whn else 'wohn', 'intra')
        for split in ['positive', 'negative']:
            sentences = load_nli(r'/home/LAB/limx/project/hyperbolic/SimCSE/data/nli_train.csv', 
                split, whn)
            collect_intra_adj_num(model, tokenizer, device,
                sentences, cache_dir, instance, dump_dir,
                max_num=10000, adj_max_distance=4., split=split)
    
    for split in ['positive', 'negative']:
        collect_intra_adj_num(model, tokenizer, device,
            load_wiki(r'/home/LAB/limx/project/hyperbolic/SimCSE/data/wiki1m_train.txt', split), 
            cache_dir, 'dropout', os.path.join(r'results/metrics/dropout', 'intra'), 
            max_num=10000, adj_max_distance=4., split=split)
        collect_intra_adj_num(model, tokenizer, device,
            load_other_augmentation(r'data/230730/wiki1m_token_shuffle_train.jsonl', split), 
            cache_dir, 'token_shuffle', os.path.join(r'results/metrics/token_shuffle', 'intra'), 
            max_num=10000, adj_max_distance=4., split=split)
        collect_intra_adj_num(model, tokenizer, device,
            load_other_augmentation(r'data/230730/wiki1m_token_cutoff_train.jsonl', split), 
            cache_dir, 'token_cutoff', os.path.join(r'results/metrics/token_cutoff', 'intra'), 
            max_num=10000, adj_max_distance=4., split=split)
    '''