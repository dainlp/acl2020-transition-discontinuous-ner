'''
Usage: the input file should be of text-inline format
python evaluate.py --gold_filepath /data/dai031/Experiments/CADEC/adr/split/test.txt --pred_filepath /data/dai031/Experiments/flair/cadec-adr/test.txt
python evaluate.py --gold_filepath /data/dai031/Experiments/CADEC/adr/split/test.txt --pred_filepath /data/dai031/Experiments/TransitionDiscontinuous/cadec-50542/test.pred
'''
import argparse, os, sys
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, os.path.abspath("../.."))
from xdai.ner.mention import Mention


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    ## Required
    parser.add_argument("--gold_filepath", default=None, type=str)
    parser.add_argument("--pred_filepath", default=None, type=str)
    args, _ = parser.parse_known_args()
    return args


'''Update: 2019-Nov-9'''
def compute_f1(TP: int, FP: int, FN: int) -> Dict:
    precision = float(TP) / float(TP + FP) if TP + FP > 0 else 0
    recall = float(TP) / float(TP + FN) if TP + FN > 0 else 0
    f1 = 2. * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    return precision, recall, f1


'''Update: 2019-Nov-9'''
def compute_on_corpus(gold_corpus: List[List[str]], pred_corpus: List[List[str]]):
    assert len(gold_corpus) == len(pred_corpus) # number of sentences

    TP, FP, FN = defaultdict(int), defaultdict(int), defaultdict(int)
    for gold_sentence, pred_sentence in zip(gold_corpus, pred_corpus):
        for gold in gold_sentence:
            if gold in pred_sentence:
                TP[gold.split()[-1]] += 1
            else:
                FN[gold.split()[-1]] += 1
        for pred in pred_sentence:
            if pred not in gold_sentence:
                FP[pred.split()[-1]] += 1

    entity_types = set(TP.keys()) | set(FP.keys()) | set(FN.keys())
    metrics = {}
    precision_per_type, recall_per_type, f1_per_type = [], [], []
    for t in entity_types:
        precision, recall, f1 = compute_f1(TP[t], FP[t], FN[t])
        metrics["%s-precision" % t] = precision
        precision_per_type.append(precision)
        metrics["%s-recall" % t] = recall
        recall_per_type.append(recall)
        metrics["%s-f1" % t] = f1
        f1_per_type.append(f1)

    metrics["macro-precision"] = sum(precision_per_type) / len(precision_per_type) if len(precision_per_type) > 0 else 0.0
    metrics["macro-recall"] = sum(recall_per_type) / len(recall_per_type) if len(recall_per_type) > 0 else 0.0
    metrics["macro-f1"] = sum(f1_per_type) / len(f1_per_type) if len(f1_per_type) > 0 else 0.0

    precision, recall, f1 = compute_f1(sum(TP.values()), sum(FP.values()), sum(FN.values()))
    metrics["micro-precision"] = precision
    metrics["micro-recall"] = recall
    metrics["micro-f1"] = f1

    return metrics


'''Update: 2019-Nov-9'''
def compute_on_sentences_with_disc(gold_corpus, pred_corpus):
    assert len(gold_corpus) == len(pred_corpus)

    gold_disc_corpus, pred_disc_corpus = [], []
    for gold_sentence, pred_sentence in zip(gold_corpus, pred_corpus):
        gold_mentions = Mention.create_mentions("|".join(gold_sentence))
        if any(m.discontinuous for m in gold_mentions):
            gold_disc_corpus.append(gold_sentence)
            pred_disc_corpus.append(pred_sentence)

    metrics = compute_on_corpus(gold_disc_corpus, pred_disc_corpus)
    return {"sentences_with_disc-%s" % k: v for k, v in metrics.items()}


'''Update: 2019-Nov-9'''
def compute_on_disc_mentions(gold_corpus, pred_corpus):
    assert len(gold_corpus) == len(pred_corpus)

    TP, FP, FN = 0.0, 0.0, 0.0
    for gold_sentence, pred_sentence in zip(gold_corpus, pred_corpus):
        gold_mentions = [m for m in Mention.create_mentions("|".join(gold_sentence)) if m.discontinuous]
        pred_mentions = [m for m in Mention.create_mentions("|".join(pred_sentence)) if m.discontinuous]
        for pred in pred_mentions:
            if str(pred) in gold_sentence:
                TP += 1
            else:
                FP += 1
        for gold in gold_mentions:
            if str(gold) not in pred_sentence:
                FN += 1

    precision, recall, f1 = compute_f1(TP, FP, FN)
    return {"disc-mention-micro-precision": precision, "disc-mention-micro-recall": recall, "disc-mention-micro-f1": f1}


if __name__ == "__main__":
    args = parse_parameters()

    sentences = []
    gold_mentions, pred_mentions = [], []
    with open(args.pred_filepath) as f:
        for sentence in f:
            sentences.append(sentence.strip())
            if not args.gold_filepath:
                gold = next(f).strip()
                gold = gold.split("|") if len(gold) > 0 else []
                gold_mentions.append(gold)
            pred = next(f).strip()
            pred = pred.split("|") if len(pred) > 0 else []
            pred_mentions.append(pred)
            assert len(next(f).strip()) == 0

    if args.gold_filepath is not None:
        with open(args.gold_filepath) as f:
            sent_id = 0
            for sentence in f:
                assert sentence.strip() == sentences[sent_id]
                sent_id += 1
                gold = next(f).strip()
                gold = gold.split("|") if len(gold) > 0 else []
                gold_mentions.append(gold)
                assert len(next(f).strip()) == 0

    metrics = compute_on_corpus(gold_mentions, pred_mentions)
    for k, v in metrics.items():
        if k.find("micro") >= 0:
            print(k, v)

    metrics = compute_on_sentences_with_disc(gold_mentions, pred_mentions)
    for k, v in metrics.items():
        if k.find("micro") >= 0:
            print(k, v)

    metrics = compute_on_disc_mentions(gold_mentions, pred_mentions)
    for k, v in metrics.items():
        if k.find("micro") >= 0:
            print(k, v)