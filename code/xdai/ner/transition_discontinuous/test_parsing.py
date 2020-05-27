from __future__ import absolute_import, division, print_function, unicode_literals
import logging, os, sys
sys.path.insert(0, os.path.abspath("../../.."))

from xdai.ner.transition_discontinuous.parsing import Parser


logger = logging.getLogger(__name__)


def _sanity_check_actions():
    sentences, actions = [], []
    sentences.append("all joints , muscles and tendons hurt all over my body")
    actions.append("OUT SHIFT OUT SHIFT OUT SHIFT SHIFT RIGHT-REDUCE COMPLETE-Finding RIGHT-REDUCE COMPLETE-Finding RIGHT-REDUCE COMPLETE-Finding OUT OUT OUT OUT")
    # "1,1,6,6 Finding|3,3,6,6 Finding|5,6 Finding"

    sentences.append("Very severe pain in arms , knees , hands .")
    actions.append("SHIFT SHIFT REDUCE SHIFT REDUCE SHIFT REDUCE SHIFT LEFT-REDUCE COMPLETE-Finding OUT SHIFT LEFT-REDUCE COMPLETE-Finding OUT SHIFT REDUCE COMPLETE-Finding OUT")
    # "0,3,6,6 ADR|0,3,8,8 ADR|0,4 ADR"

    parse = Parser()
    for s, a in zip(sentences, actions):
        mentions = parse.parse(a.split(), len(s.split()))
        logger.info("|".join([str(m) for m in mentions]))


def _sanity_check_mention2actions():
    sentences, mentions = [], []
    sentences.append("could hardly walk or lift my arms")
    mentions.append("1,1,4,6 ADR|1,2 ADR")
    # ['OUT', 'SHIFT', 'SHIFT', 'LEFT-REDUCE', 'COMPLETE-ADR', 'OUT', 'SHIFT', 'REDUCE', 'SHIFT', 'REDUCE', 'SHIFT', 'REDUCE', 'COMPLETE-ADR']


    sentences.append("all joints , muscles and tendons hurt all over my body")
    mentions.append("1,1,6,6 Finding|3,3,6,6 Finding|5,6 Finding")
    # "OUT SHIFT OUT SHIFT OUT SHIFT SHIFT RIGHT-REDUCE COMPLETE-Finding RIGHT-REDUCE COMPLETE-Finding RIGHT-REDUCE COMPLETE-Finding OUT OUT OUT OUT"

    sentences.append("Very severe pain in arms , knees , hands .")
    mentions.append("0,3,6,6 ADR|0,3,8,8 ADR|0,4 ADR")
    # "SHIFT SHIFT REDUCE SHIFT REDUCE SHIFT REDUCE SHIFT LEFT-REDUCE COMPLETE-Finding OUT SHIFT LEFT-REDUCE COMPLETE-Finding OUT SHIFT REDUCE COMPLETE-Finding OUT"

    parse = Parser()
    for i, s in enumerate(sentences):
        actions = parse.mention2actions(mentions[i], len(s.split()))
        logger.info(actions)


def _sanity_check_instance(sentence, mentions, verbose=False):
    sentence_length = len(sentence.split())

    parse = Parser()

    actions = parse.mention2actions(mentions, sentence_length)
    preds = parse.parse(actions, sentence_length)
    golds = mentions.split("|")
    preds = [str(m) for m in preds]

    FP, FN = 0, 0
    for pred in preds:
        if pred not in golds:
            FP += 1
    for gold in golds:
        if gold not in preds:
            FN += 1

    if verbose and (not (FP == 0 and FN == 0)):
        logger.info(sentence)
        logger.info(golds)
        logger.info(preds)

    return FP == 0 and FN == 0


def check_dataset(data_dir):
    for split in ["train", "dev", "test"]:
        total_sentences, error_sentences = 0, 0
        if not os.path.exists(os.path.join(data_dir, "%s.txt" % split)): continue
        with open(os.path.join(data_dir, "%s.txt" % split)) as f:
            for line in f:
                sentence = line.strip()
                if len(sentence) == 0: continue
                mentions = next(f).strip()
                if len(mentions) > 0:
                    correct = _sanity_check_instance(sentence, mentions, verbose=True)
                    total_sentences += 1
                    if not correct:
                        error_sentences += 1
                assert next(f).strip() == ""
        logger.info("%d errors out of %d sentences" % (error_sentences, total_sentences))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, filename=None)

    # _sanity_check_actions()
    #_sanity_check_mention2actions()
    #check_dataset("/data/dai031/Experiments/ShARe2013")
    check_dataset("/data/dai031/Experiments/CADEC/adr/split")