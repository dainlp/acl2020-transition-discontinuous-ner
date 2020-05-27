'''Update date: 2020-Jan-13'''
import argparse
from collections import defaultdict


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    ## Required
    parser.add_argument("--input_ann", default="/data/dai031/Experiments/CADEC/tokens.ann", type=str)
    parser.add_argument("--input_tokens", default="/data/dai031/Experiments/CADEC/tokens", type=str)
    parser.add_argument("--output_filepath", default="/data/dai031/Experiments/CADEC/text-inline", type=str)
    parser.add_argument("--no_doc_info", action="store_true")

    args, _ = parser.parse_known_args()
    return args


def output_sentence(f, tokens, mentions, doc=None):
    def check_mention_text(tokens, mentions):
        for mention in mentions:
            tokenized_mention = []
            indices = [int(i) for i in mention[0].split(",")]
            for i in range(0, len(indices), 2):
                start, end = indices[i], indices[i + 1]
                tokenized_mention += tokens[start:end + 1]

            if "".join(tokenized_mention) != mention[2].replace(" ", ""):
                print("%s (original) vs %s (tokenized)" % (mention[2], " ".join(tokenized_mention)))

    if doc is not None:
        f.write("Document: %s\n" % doc)
    f.write("%s\n" % (" ".join(tokens)))
    check_mention_text(tokens, mentions)
    mentions = ["%s %s" % (m[0], m[1]) for m in mentions]
    f.write("%s\n\n" % ("|".join(mentions)))


def load_mentions(filepath):
    mentions = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            sp = line.strip().split("\t")
            assert len(sp) == 5
            doc, sent_idx, label, indices, mention = sp
            mentions[(doc, int(sent_idx))].append((indices, label, mention))
    return mentions


if __name__ == "__main__":
    args = parse_parameters()
    mentions = load_mentions(args.input_ann)

    with open(args.output_filepath, "w") as out_f:
        with open(args.input_tokens) as in_f:
            pre_doc, sent_idx = None, 0
            tokens = []
            for line in in_f:
                if len(line.strip()) == 0:
                    if len(tokens) > 0:
                        assert pre_doc is not None
                        output_sentence(out_f, tokens, mentions.get((pre_doc, sent_idx), []),
                                        None if args.no_doc_info else pre_doc)
                        sent_idx += 1
                        tokens = []
                    continue
                sp = line.strip().split()
                token, doc, _, _ = sp
                if pre_doc is None:
                    pre_doc = doc
                if pre_doc != doc:
                    pre_doc = doc
                    sent_idx = 0
                    assert len(tokens) == 0
                tokens.append(token)
            if len(tokens) > 0:
                assert pre_doc is not None
                output_sentence(out_f, tokens, mentions.get((pre_doc, sent_idx), []),
                                None if args.no_doc_info else pre_doc)