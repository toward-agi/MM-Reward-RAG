import json 
from pathlib import Path

# 生成一个.temp文件夹，然后随机生成一个路径，tsv结尾

# with open(out_path, 'w') as f:
#     f.write('\t'.join(header) + '\n')
#     for item in data:
#         # use json.dumps(my_list) to convert list to string
#         values = [str(item['Guid']), str(item['Qcate']), str(item['Q']), json.dumps(item['A']), str(item['Output_conf']), json.dumps(item['Output'])]
#         f.write('\t'.join(values) + '\n')

"""
  "Q": original_data["Q"].strip('"'),  # Remove outer quotes
                "A": [a.strip('"') for a in original_data["A"]],  # Clean quotes from answers
"""


###

import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer
from typing import List
import numpy as np


class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        # self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))


import numpy as np
import os
import json

from tqdm import tqdm
from datetime import datetime
from pytz import timezone
from word2number import w2n
import string, re
from collections import Counter
from pprint import pprint
import spacy
print("Loading spacy...")
nlp = spacy.load("en_core_web_sm", disable=["ner","textcat","parser"])
np.set_printoptions(precision=4)

import argparse

import sys
sys.path.append("/home/longfei/BARTScore")
# from bart_score import BARTScorer

print("Loading BART scorer...")
bart_scorer_ParaBank = BARTScorer(device='cuda:0', checkpoint='/data/zhiyuan/mdoels/bart-large-cnn')
bart_scorer_ParaBank.load(path='/data/longfei/bart.pth') # Please change the path to bart.pth


def detectNum(l):
    result = []
    for w in l:
        try: result.append(str(int(w)))
        except: pass
    return result
def toNum(word):
    if word == 'point': return word
    try: return w2n.word_to_num(word)
    except:
        return word

def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text): # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])

    def remove_punc(text):
        exclude = set(string.punctuation) - set(['.'])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1) # remove '.' if it's not a decimal point

    def lower(text):
        return text.lower()
    
    def lemmatization(text):
        return " ".join([token.lemma_ for token in nlp(text)])

    if len(s.strip()) == 1:
        # accept article and punc if input is a single char
        return white_space_fix(lower(s))
    elif len(s.strip().split()) == 1: 
        # accept article if input is a single word
        return lemmatization(white_space_fix(remove_punc(lower(s))))

    return lemmatization(white_space_fix(remove_articles(remove_punc(lower(s)))))

# VQA Eval (SQuAD style EM, F1)
def compute_vqa_metrics(cands, a, exclude="", domain=None):
    if len(cands) == 0: return (0,0,0)
    bow_a = normalize_text(a).split()
    F1 = []
    EM = 0
    RE = []
    PR = []
    e = normalize_text(exclude).split()
    for c in cands:
        bow_c = [w for w in normalize_text(c).split() if not w in e]
        if domain == {"NUMBER"}: 
            bow_c = detectNum(bow_c)
            bow_a = detectNum(bow_a)
        elif domain is not None: 
            bow_c = list(domain.intersection(bow_c))
            bow_a = list(domain.intersection(bow_a))
        
        #print(bow_c)
        #print(bow_a)
        if bow_c == bow_a:
            EM = 1
        common = Counter(bow_a) & Counter(bow_c)
        num_same = sum(common.values())
        if num_same == 0:
            return (0,0,0,0,0)
        precision = 1.0 * num_same / len(bow_c)
        recall = 1.0 * num_same / len(bow_a)
        RE.append(recall)
        PR.append(precision)

        f1 = 2*precision*recall / (precision + recall + 1e-5)
        F1.append(f1)
    
    PR_avg = np.mean(PR)
    RE_avg = np.mean(RE)
    F1_avg = np.mean(F1)
    F1_max = np.max(F1)
    return (F1_avg, F1_max, EM, RE_avg, PR_avg)

color_set= {'orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite', 'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey', 'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet', 'transparent', 'yes', 'silver', 'chrome', 'green', 'aqua'}
shape_set = {'globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross', 'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube', 'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'h', 'cuboid', 'step', 'rectangle', 'dot', 'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape', 'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight', 'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical', 'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'}
yesno_set = {'yes', 'no'}


TABLE = str.maketrans(dict.fromkeys(string.punctuation)) 
def normalize_text_for_bart(x): # Light text normalization for WebQA eval: white space fix + punctuation removal
    return " ".join(x.translate(TABLE).split())

def compute_bartscore_ParaBank(c, a, switch=False):
    c_removepunc = [normalize_text_for_bart(x) for x in c]
    a_removepunc = [normalize_text_for_bart(x) for x in a]
    if switch: score = np.exp(bart_scorer_ParaBank.score(c_removepunc, a_removepunc))
    else: score = np.exp(bart_scorer_ParaBank.score(a_removepunc, c_removepunc))
    return score


def _webqa_acc_approx(predction, ground_truth, domain=None):
    """VQA Eval (SQuAD style EM, F1)"""
    bow_pred = normalize_text(predction).split()
    bow_target = normalize_text(ground_truth).split()
    if domain == {"NUMBER"}:
        bow_pred = detectNum(bow_pred)
        bow_target = detectNum(bow_target)
    elif domain is not None:
        bow_pred = list(domain.intersection(bow_pred))
        bow_target = list(domain.intersection(bow_target))
    else:
        # TODO: fine-grained evaluation (e.g., content words) for text question types
        bow_pred = bow_pred
        bow_target = bow_target

    common = Counter(bow_target) & Counter(bow_pred)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = num_same / len(bow_pred)
    recall = num_same / len(bow_target)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, recall, precision


def webqa_metrics_approx(prediction, ground_truth, Qcate="text"):
    f1, recall, precision = _webqa_acc_approx(
        prediction,
        ground_truth,
        domain={
            "color": color_set,
            "shape": shape_set,
            "YesNo": yesno_set,
            "number": {"NUMBER"},
            "text": None,
            "Others": None,
            "choose": None,
        }[Qcate],
    )
    return f1, recall, precision
    # if Qcate in ["color", "shape", "number", "YesNo"]:
    #     accuracy = f1
    # else:
    #     accuracy = recall
    # return {"acc_approx": accuracy}

def eval_webqa(results_file_path, experimental_name):

    parser = argparse.ArgumentParser()
    parser.add_argument("--Qcate_breakdown", type=str, default='["all"]')
    parser.add_argument("--file", type=str, default='result.tsv')
    parser.add_argument('--no_norm', action='store_true')
    #parser.add_argument('--dir', type=str, default='/data/track/mm-RewardRAG/results/webqa/vanilla_lvlm')
    parser.add_argument('--dir', type=str, default='/data/track/mm-RewardRAG/eval_utils/generation/results/webqa')
    parser.add_argument('--output_idx', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    args.file = f"{experimental_name}.tsv"
    os.makedirs("/data/track/mm-RewardRAG/.temp", exist_ok=True)
    import random
    random_path = os.path.join("/data/track/mm-RewardRAG/.temp", str(random.randint(0, 100000)) + ".tsv")
    out_path = random_path

    header = ['Guid', 'Qcate', 'Q', 'A', 'Output_conf', 'Output']
    val_set = json.load(open("/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json"))

    results_file = [
        json.loads(line)
        for line in open(results_file_path, 'r')
    ]

    id_to_output = {
        item["data_id"]: item["answer"]
        for item in results_file
    }

    with open(out_path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for id in val_set.keys():
            if id not in id_to_output:
                print(f"Missing response for {id}")
                continue
            item = val_set[id]
            values = [str(item['Guid']), str(item['Qcate']), str(item['Q'].strip('"')), json.dumps([a.strip('"') for a in item["A"]]), str([]), json.dumps([id_to_output[id]])]
            f.write('\t'.join(values) + '\n')

    Qcate_breakdown = json.loads(args.Qcate_breakdown)
    print("Use categories: ", Qcate_breakdown)
    print("Use normalization = ", not args.no_norm)
    print("Output_idx = ", args.output_idx)
    # Please change the path to your output folder
    # with open(os.path.join(args.dir, args.file), "r") as fp: # ok这里是联合组成一个路径就行；
    with open(out_path, "r") as fp: # ok这里是联合组成一个路径就行；
        lines = fp.readlines()
        header = lines[0].strip().split('\t')
        rows = lines[1:]
    key = dict(zip(header, range(len(header))))
    pprint(key)
    F1_avg_scores = {'All':[], 'number':[], 'YesNo':[], 'choose':[], 'color':[], 'shape':[], 'Others':[], 'text':[]}
    F1_max_scores = {'All':[], 'number':[], 'YesNo':[], 'choose':[], 'color':[], 'shape':[], 'Others':[], 'text':[]}
    EM_scores = {'All':[], 'number':[], 'YesNo':[], 'choose':[], 'color':[], 'shape':[], 'Others':[], 'text':[]}
    RE_scores = {'All':[], 'number':[], 'YesNo':[], 'choose':[], 'color':[], 'shape':[], 'Others':[], 'text':[]}
    PR_scores= {'All':[], 'number':[], 'YesNo':[], 'choose':[], 'color':[], 'shape':[], 'Others':[], 'text':[]}
    fluency_scores = {'All':[], 'number':[], 'YesNo':[], 'choose':[], 'color':[], 'shape':[], 'Others':[], 'text':[]}
    acc_scores = {'All':[], 'number':[], 'YesNo':[], 'choose':[], 'color':[], 'shape':[], 'Others':[], 'text':[]}
    mul_scores = {'All':[], 'number':[], 'YesNo':[], 'choose':[], 'color':[], 'shape':[], 'Others':[], 'text':[]}


    output_Q = []
    output_A = []
    output_N = []
    output_O = []
    #output_KA = []
    output_G = []
    output_QC = []
    # Guid	Qcate	Q	A	Keywords_A	Output_conf	Output
    for r in tqdm(rows):
        
        datum = r.strip().split('\t')
        Qcate = datum[key['Qcate']]
        if (not 'all' in Qcate_breakdown) and (not Qcate in Qcate_breakdown): continue
        O = json.loads(datum[key['Output']])
        C = [O[args.output_idx]]
        #Keywords_A = datum[key['Keywords_A']]
        A = json.loads(datum[key['A']])
        #normalizer = guid2norm[datum[key['Guid']]]
        normalizer = compute_bartscore_ParaBank(A, A)
        
        output_Q.append(datum[key['Q']])
        output_A.append(A)
        output_N.append(normalizer)
        output_O.append(O)
        #output_KA.append(Keywords_A)
        output_G.append(datum[key['Guid']])
        output_QC.append(Qcate)
        
        if args.no_norm:
            score = min(1, np.max(compute_bartscore_ParaBank(C*len(A), A)))
        else: score = min(1, np.max(compute_bartscore_ParaBank(C*len(A), A)/np.array(normalizer)))

        if Qcate == 'color': F1_avg, RE_avg, PR_avg = webqa_metrics_approx(C[0], A[0], "color")
        elif Qcate == 'shape': F1_avg, RE_avg, PR_avg = webqa_metrics_approx(C[0], A[0], "shape")
        elif Qcate == 'YesNo': F1_avg, RE_avg, PR_avg = webqa_metrics_approx(C[0], A[0], "YesNo")
        elif Qcate == 'number': F1_avg, RE_avg, PR_avg = webqa_metrics_approx(C[0], A[0], "number")
        else: F1_avg, RE_avg, PR_avg = webqa_metrics_approx(C[0], A[0])
        fluency_scores['All'].append(score)
        fluency_scores[Qcate].append(score)
        if Qcate in ['color', 'shape', 'number', 'YesNo']: 
            acc_scores['All'].append(F1_avg)
            acc_scores[Qcate].append(F1_avg)
            mul_scores['All'].append(F1_avg * score)
            mul_scores[Qcate].append(F1_avg * score)
        else: 
            acc_scores['All'].append(RE_avg)
            acc_scores[Qcate].append(RE_avg)
            mul_scores['All'].append(RE_avg * score)
            mul_scores[Qcate].append(RE_avg * score)

        F1_avg_scores['All'].append(F1_avg)
        # F1_max_scores['All'].append(F1_max)
        # EM_scores['All'].append(EM)
        RE_scores['All'].append(RE_avg)
        PR_scores['All'].append(PR_avg)
        
        F1_avg_scores[Qcate].append(F1_avg)
        # F1_max_scores[Qcate].append(F1_max)
        # EM_scores[Qcate].append(EM)
        RE_scores[Qcate].append(RE_avg)
        PR_scores[Qcate].append(PR_avg)

    assert len(F1_avg_scores) == len(RE_scores) == len(PR_scores) == len(fluency_scores) == len(acc_scores) == len(mul_scores)
    assert len(output_Q) == len(output_A) == len(output_N) == len(output_O) ==len(output_G) == len(output_QC) == len(mul_scores['All'])
                        
    print("#eval samples = ", len(mul_scores['All']))
    F1_avg = np.mean(F1_avg_scores['All'])
    # F1_max = np.mean(F1_max_scores['All'])
    # EM = np.mean(EM_scores['All'])
    RE_avg = np.mean(RE_scores['All'])
    PR_avg = np.mean(PR_scores['All'])

    fluency_avg = np.mean(fluency_scores['All'])
    acc_avg = np.mean(acc_scores['All'])
    mul_avg = np.mean(mul_scores['All'])

    print("F1_avg = {}".format(F1_avg))
    #print("F1_max = {}".format(F1_max))
    #print("EM = {}".format(EM))
    print("RE_avg = {}".format(RE_avg))
    #print("PR_avg = {}".format(PR_avg))
    print("fluency_avg = {}".format(fluency_avg))
    print("acc_avg = {}".format(acc_avg))
    print("mul_avg = {}".format(mul_avg))
    print(" ------------------------------------------------------------------------------------ \n")
    if not 'all' in Qcate_breakdown: args.file = args.file.split(".")[0] + "_{}".format("|".join(Qcate_breakdown)) + ".tsv"

    if args.output_idx > 0: args.file = args.file.replace(".tsv", "_{}.tsv".format(args.output_idx))

    with open(os.path.join(args.dir, args.file.replace(".tsv", ".txt")), "w") as f:
        f.write(datetime.now(tz=timezone('US/Eastern')).strftime("%y-%m-%d %H:%M:%S") + '\n')
        f.write('\nUse Q categories: {}\nUse normalization = {}\n#Eval_samples = {}\n'.format(Qcate_breakdown, not args.no_norm, len(mul_scores['All'])))
        
        f.write('\n --------------------- metrics -----------------------\n')
        for k in ['All', 'YesNo', 'choose', 'color', 'shape', 'number', 'Others', 'text']:
            f.write("\n--- Category = {}: ---\n".format(k))
            f.write('\n'.join(["F1_avg = {}".format(np.mean(F1_avg_scores[k])), "EM = {}".format(np.mean(EM_scores[k]))]))
            f.write('\n\n')
            f.write('\n'.join(["RE_avg = {}".format(np.mean(RE_scores[k])), "PR_avg = {}".format(np.mean(PR_scores[k]))]))
            f.write('\n\n')
            f.write('\n'.join(["fluency_avg = {}".format(np.mean(fluency_scores[k])), "acc_avg = {}".format(np.mean(acc_scores[k])), "mul_avg = {}".format(np.mean(mul_scores[k]))]))
            f.write('\n\n')
        f.write('-----Starting writing results:-----')

        for guid, qcate, q, a, n, o, re, f1, b, acc, m in zip(output_G, output_QC, output_Q, output_A, output_N, output_O, RE_scores['All'], F1_avg_scores['All'], fluency_scores['All'], acc_scores['All'], mul_scores['All']):
            f.write("\n\n")
            f.write("\n".join(['Guid === {}\n Qcate === {}'.format(guid, qcate), 
                'Q === {}'.format(q), '\n'.join(a), 
                'Normalizer === {}\n'.format(n), '\n'.join(o), 
                'RE === {}'.format(re), 'F1 === {}'.format(f1),
                'Fluency === {}'.format(b),
                'Accuracy === {}'.format(acc),
                "mul === {}".format(m)]))

    # 删掉那个临时文件
    os.remove(random_path)

def test():
    eval_webqa("/data/track/mm-RewardRAG/experiments/5-answer/w_all_hn/results/3b/webqa/answers_with_gt.jsonl","qwen25_w_gt_prompt_version_0503_with_all_hn")

if __name__ == "__main__":
    test()
