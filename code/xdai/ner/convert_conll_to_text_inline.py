import argparse, os, sys
sys.path.insert(0, os.path.abspath("../.."))

from xdai.ner.mention import bio_tags_to_mentions, bioes_to_bio


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    ## Required
    parser.add_argument("--input_filepath", default="/data/dai031/Experiments/flair/conll2003/test.tsv", type=str)
    parser.add_argument("--output_filepath", default="/data/dai031/Experiments/flair/conll2003/test.txt", type=str)
    parser.add_argument("--pred_column_idx", default=-1, type=int)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_parameters()

    sentences = []
    with open(args.input_filepath) as f:
        tokens, tags = [], []
        for line in f:
            sp = line.strip().split()
            if len(sp) < 2 or sp[0] == "-DOCSTART-":
                if len(tokens) > 0:
                    sentences.append((tokens, bio_tags_to_mentions(bioes_to_bio(tags))))
                tokens, tags = [], []
                continue
            tokens.append(sp[0])
            tags.append(sp[args.pred_column_idx])
        if len(tokens) > 0:
            sentences.append((tokens, bio_tags_to_mentions(bioes_to_bio(tags))))

    with open(args.output_filepath, "w") as f:
        for (tokens, mentions) in sentences:
            f.write("%s\n" % " ".join(tokens))
            mentions = [str(m) for m in mentions]
            f.write("%s\n" % "|".join(mentions))
            f.write("\n")