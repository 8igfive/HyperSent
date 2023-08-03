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

def load_wiki(txt_path: str, mode: str = 'positive'):
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

def load_nli(csv_path: str, mode: str = 'positive'):
    df = pd.read_csv(csv_path)
    lines = []
    for row in df.iloc:
        lines.append((row['sent0'], row['sent1'], row['hard_neg']))
    
    if mode == 'positive':
        return [(line[0], line[1]) for line in lines]
    elif mode == 'negative':
        return [(line[0], line[2]) for line in lines]
    elif mode == 'all':
        res = []
        for line in lines:
            res.append(line[0])
            res.append(line[1])
            res.append(line[2])
        return res
    else:
        raise NotImplementedError

def load_stsb(split: str, mode: str = 'positive'):
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

def load_aigen(path: str, mode: str = 'positive'):
    with open(path, 'r', encoding='utf8') as fi:
        rows = [json.loads(row) for row in fi]
        aigens = [tuple(row[key] for key in ['sentence', '5', '4', '3', '2', '1', '0'] if row[key] != '') 
                  for row in rows if row['split'] == 'aigen']
        others = [row['sentence'] for row in rows if row['split'] == 'other']
    
    if mode == 'positive':
        # [(row[0], row[1]) for row in aigens] + [(row, row) for row in others]
        return [(row[0], row[1]) for row in aigens]
    elif mode == 'negative':
        another_others = others.copy()
        random.shuffle(another_others)
        # [(row[0], row[-1]) for row in aigens] + [(row0, row1) for row0, row1 in zip(others, another_others)]
        return [(row[0], row[-1]) for row in aigens]
    elif mode == 'all':
        res = []
        for row in aigens:
            for sentence in row:
                res.append(sentence)
        for row in others:
            res.append(row)
        return res
    else:
        raise NotImplementedError

def load_other_augmentation(path: str, mode: str = 'positive'):
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

def collect_embeddings(ckpt_dir, sentences, max_size=None, base_model: str = 'bert-base-uncased'):
    save_name = 'ckpts_features_paired' if isinstance(sentences[0], tuple) else 'ckpts_features'
    if max_size is not None:
        save_name = f'{save_name}_{max_size}.pt'
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

def cal_alignment_and_uniformity(ckpt_dir, corpus_path, load_fn, 
    cal_aligment=True, cal_uniformity=True, 
    alignment_max_size=None, uniformity_max_size=None):

    # alignment
    if cal_aligment:
        print('Calculate Alignment:\n')
        sentences = load_fn(corpus_path, mode='positive')
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
        sentences = load_fn(corpus_path, mode='all')
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

def collect_metrics4corpus(load_fn, corpus_arg, mode='positive', metrics='mer'):
    sentence_pairs = load_fn(corpus_arg, mode=mode)
    dump_dir = os.path.join(r'result/metrics', load_fn.__name__.split('_')[-1])
    os.makedirs(dump_dir, exist_ok=True)

    if metrics == 'mer':
        ms = cal_mers(sentence_pairs)
    elif metrics == 'iou':
        ms = cal_ious(sentence_pairs)
    else:
        raise NotImplementedError
    
    with open(os.path.join(dump_dir, f'{mode}_{metrics}'), 'w', encoding='utf8') as fo:
        fo.write('\n'.join(str(m) for m in ms))

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
    
    cal_alignment_and_uniformity(r'results/runs/test_sup_aigen_nli_whn',
        r'validation', load_stsb,  
        alignment_max_size=10000, uniformity_max_size=10000)
    cal_alignment_and_uniformity(r'results/runs/test_sup_aigen_nli_wohn',
        r'validation', load_stsb,  
        alignment_max_size=10000, uniformity_max_size=10000)
    cal_alignment_and_uniformity(r'results/runs/test_sup_aigen_sts_whn',
        r'validation', load_stsb,  
        alignment_max_size=10000, uniformity_max_size=10000)
    cal_alignment_and_uniformity(r'results/runs/test_sup_aigen_sts_wohn',
        r'validation', load_stsb,  
        alignment_max_size=10000, uniformity_max_size=10000)

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

    assemble_alignment_and_uniformity(r'results/runs/test_sup_aigen_nli_whn')
    assemble_alignment_and_uniformity(r'results/runs/test_sup_aigen_nli_wohn')
    assemble_alignment_and_uniformity(r'results/runs/test_sup_aigen_sts_whn')
    assemble_alignment_and_uniformity(r'results/runs/test_sup_aigen_sts_wohn')

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
        

    # cal_anu_bbu('bert-base-uncased', r'/home/LAB/limx/project/hyperbolic/SimCSE/data/wiki1m_heldout.txt',
    #             load_wiki, alignment_max_size=10000, uniformity_max_size=10000)
    # cal_anu_bbu('bert-base-uncased', r'/home/LAB/limx/project/hyperbolic/SimCSE/data/nli_heldout.csv',
    #             load_nli, alignment_max_size=10000, uniformity_max_size=10000)
