import argparse, logging, os, sys
from collections import defaultdict

logger = logging.getLogger(__name__)


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--log_filepath", default="output.log", type=str)

    args, _ = parser.parse_known_args()
    return args


def _output_sentence(out_f, tokens, anns, document, sentence):
    out_f.write("%s\n" % (" ".join(tokens)))
    mentions = anns.get((document, sentence), [])
    mentions = ["%s Disorder" % (mention) for mention in mentions]
    out_f.write("%s\n\n" % ("|".join(mentions)))


def read_data(ann_filepath, tokens_filepath):
    anns = defaultdict(list)
    with open(ann_filepath) as f:
        for line in f:
            sp = line.strip().split("\t")
            assert len(sp) == 4 or len(sp) == 5
            document, sentence_idx, _, indices = sp[0:4]
            anns[(document, int(sentence_idx))].append((indices))

    with open(tokens_filepath) as f:
        sentences = []
        pre_doc, sentence_idx = None, 0
        tokens = []
        for line in f:
            if len(line.strip()) == 0:
                if len(tokens) > 0:
                    assert pre_doc is not None
                    sentences.append((pre_doc, sentence_idx, tokens))
                    sentence_idx += 1
                    tokens = []
                continue
            sp = line.strip().split()
            token, cur_doc, _, _ = sp
            if pre_doc is None:
                pre_doc = cur_doc
            if pre_doc != cur_doc:
                pre_doc = cur_doc
                sentence_idx = 0
                assert len(tokens) == 0
            tokens.append(token)
        if len(tokens) > 0:
            assert pre_doc is not None
            sentences.append((pre_doc, sentence_idx, tokens))

    return anns, sentences


if __name__ == "__main__":
    args = parse_parameters()
    handlers = [logging.FileHandler(filename=args.log_filepath), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, handlers=handlers)

    train_ann, train_sentences = read_data("train.token.ann", "train.tokens")
    test_ann, test_sentences = read_data("test.token.ann", "test.tokens")

    dev_list = [n.strip() for n in open("dev.list").readlines()]

    with open(os.path.join(args.output_dir, "train.txt"), "w") as f:
        for sentence in train_sentences:
            document, sentence_idx, tokens = sentence
            if document not in dev_list:
                _output_sentence(f, tokens, train_ann, document, sentence_idx)

    with open(os.path.join(args.output_dir, "dev.txt"), "w") as f:
        for sentence in train_sentences:
            document, sentence_idx, tokens = sentence
            if document in dev_list:
                _output_sentence(f, tokens, train_ann, document, sentence_idx)

    with open(os.path.join(args.output_dir, "test.txt"), "w") as f:
        for sentence in test_sentences:
            document, sentence_idx, tokens = sentence
            _output_sentence(f, tokens, test_ann, document, sentence_idx)