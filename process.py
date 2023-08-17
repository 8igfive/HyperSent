import os
import json
import random
import pandas as pd

def combine_dataset(ratio: float, dump_path: str, 
        nli_path: str = r'data/230803/allnli_0k_2.jsonl', 
        wiki_path: str = r'data/wiki1m_aigen_0_0.jsonl'):
    assert ratio >= 0 and ratio <= 1
    with open(nli_path, 'r', encoding='utf8') as fi:
        nli = [json.loads(line) for line in fi]
        random.shuffle(nli)
    with open(wiki_path, 'r', encoding='utf8') as fi:
        wiki = [json.loads(line) for line in fi]
        random.shuffle(wiki)
    nli_len = round(len(nli) * ratio)
    wiki_len = len(nli) - nli_len
    res = []
    for line in nli[:nli_len]:
        new_line = line.copy()
        new_line['split'] = 'aigen'
        res.append(new_line)
    for line in wiki[:wiki_len]:
        res.append(line)
    
    with open(dump_path, 'w', encoding='utf8') as fo:
        fo.write('\n'.join(json.dumps(line, ensure_ascii=False) for line in res))

def get_nli_others(aigen_path: str, nli_path: str, dump_path: str):
    nli = pd.read_csv(nli_path)
    sents = [line['sent0'] for line in nli.iloc]
    sents_set = set(sents)
    with open(aigen_path, 'r', encoding='utf8') as fi:
        lines = [json.loads(line) for line in fi]
        aigens = [line for line in lines if line['split'] == 'aigen' and line['sentence'] in sents_set]
    aigens_set = set(aigen['sentence'] for aigen in aigens)
    others = [
        {
            'split': 'other',
            'sentence': sent,
            '5': '', '4': '', '3': '', '2': '', '1': '', '0': ''
        }
        for sent in sents if sent not in aigens_set
    ]

    with open(dump_path, 'w', encoding='utf8') as fo:
        fo.write('\n'.join(json.dumps(line, ensure_ascii=False) for line in (aigens + others)))

def cut_aigen_numbert(path: str, number: int, dump_path: str):
    with open(path, 'r', encoding='utf8') as fi:
        lines = [json.loads(line) for line in fi]
        aigens = [line for line in lines if line['split'] == 'aigen']
        others = [line for line in lines if line['split'] == 'other']
        assert len(aigens) + len(others) == len(lines)
    new_aigens = aigens[:number]
    new_others = others
    for aigen in aigens[number:]:
        new_aigen = aigen.copy()
        new_aigen['split'] = 'other'
        for key in range(6):
            assert str(key) in new_aigen
            new_aigen[str(key)] = ''
        new_others.append(new_aigen)
    
    with open(dump_path, 'w', encoding='utf8') as fo:
        fo.write('\n'.join(json.dumps(line, ensure_ascii=False) for line in (new_aigens + new_others)))
    
def find_common(wiki_nli_path: str, wiki_sts_path: str, 
    nli_nli_path: str, nli_sts_path: str, 
    sentence_len: int = 20, show_num: int = 3):
    with open(wiki_nli_path, 'r', encoding='utf8') as fi:
        wiki_nli = [json.loads(line) for line in fi]
        wiki_nli = [line for line in wiki_nli if line['split'] == 'aigen']
        s2wiki_nli = {line['sentence']: line for line in wiki_nli}
    
    with open(wiki_sts_path, 'r', encoding='utf8') as fi:
        wiki_sts = [json.loads(line) for line in fi]
        wiki_sts = [line for line in wiki_sts if line['split'] == 'aigen']
        s2wiki_sts = {line['sentence']: line for line in wiki_sts}

    with open(nli_nli_path, 'r', encoding='utf8') as fi:
        nli_nli = [json.loads(line) for line in fi]
        nli_nli = [line for line in nli_nli if line['split'] == 'aigen']
        s2nli_nli = {line['sentence']: line for line in nli_nli}

    with open(nli_sts_path, 'r', encoding='utf8') as fi:
        nli_sts = [json.loads(line) for line in fi]
        nli_sts = [line for line in nli_sts if line['split'] == 'aigen']
        s2nli_sts = {line['sentence']: line for line in nli_sts}

    count = 0
    print('=' * 20)
    for sent in s2wiki_nli:
        if len(sent.split()) == sentence_len and sent in s2wiki_sts:
            for key in ['sentence', '5', '4']:
                print(s2wiki_nli[sent][key])
            print('-' * 20)
            for key in ['sentence', '5', '4', '3']:
                print(s2wiki_sts[sent][key])
            print('=' * 20)
            count += 1
            if count == show_num:
                break
    
    count = 0
    print('\n\n\n' + '=' * 20)
    for sent in s2nli_nli:
        if len(sent.split()) == sentence_len and sent in s2nli_sts:
            for key in ['sentence', '5', '4']:
                print(s2nli_nli[sent][key])
            print('-' * 20)
            for key in ['sentence', '5', '4', '3']:
                print(s2nli_sts[sent][key])
            print('=' * 20)
            count += 1
            if count == show_num:
                break

if __name__ == '__main__':
    random.seed(0)
    # for i in range(11):
    #     combine_dataset(0.1 * i, f'data/230804/ratios/nli_wiki_{0.1 * i}.jsonl')

    # get_nli_others(r'data/230802/nli20k_aigen_new_nli_2.json', 
    #                r'/home/LAB/limx/project/hyperbolic/SimCSE/data/nli_for_simcse.csv',
    #                r'data/230807/nliunsup_aigen_nli_2_20k.jsonl')
    
    # get_nli_others(r'data/230802/nli20k_aigen_sts_3.json', 
    #                r'/home/LAB/limx/project/hyperbolic/SimCSE/data/nli_for_simcse.csv',
    #                r'data/230807/nliunsup_aigen_sts_3_20k.jsonl')

    # get_nli_others(r'data/230804/allnli_aigen_sts_100k_3.jsonl', 
    #                r'/home/LAB/limx/project/hyperbolic/SimCSE/data/nli_for_simcse.csv',
    #                r'data/230807/nliunsup_aigen_sts_3_100k.jsonl')

    # for number in [17500, 15000, 12500, 7500]: # 10000, 5000, 2500, 1250, 625, 1
    #     cut_aigen_numbert(r'data/230728/wiki1m_aigen_remove_negative_20k_3.json', number,
    #          os.path.join(r'data/230813', f'wiki_sts_{number}.jsonl'))
    

    find_common(r'data/230801/aigen_nli_2_230729_align.json',
                r'data/230728/wiki1m_aigen_remove_negative_20k_3.json',
                r'data/230807/nliunsup_aigen_nli_2_20k.jsonl',
                r'data/230807/nliunsup_aigen_sts_3_20k.jsonl')