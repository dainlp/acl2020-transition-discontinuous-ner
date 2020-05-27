'''Update date: 2020-Jan-13'''
import argparse, os, sys, re
from typing import List

def extract_indices_from_brat_annotation(indices: str) -> List[int]:
    indices = re.findall(r"\d+", indices)
    indices = sorted([int(i) for i in indices])
    return indices


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    ## Required
    parser.add_argument("--input_ann", default="/data/dai031/Corpora/CADEC/cadec/original-fixed", type=str)
    parser.add_argument("--input_text", default="/data/dai031/Corpora/CADEC/cadec/text", type=str)
    parser.add_argument("--output_filepath", default="/data/dai031/Experiments/CADEC/ann", type=str)

    args, _ = parser.parse_known_args()
    return args


def _get_mention_from_text(text, indices):
    tokens = []
    for i in range(0, len(indices), 2):
        tokens.append(text[int(indices[i]):int(indices[i + 1])])
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return " ".join(tokens)


if __name__ == "__main__":
    args = parse_parameters()
    num_mentions = 0

    with open(args.output_filepath, "w") as out_f:
        for doc in os.listdir(args.input_ann):
            with open(os.path.join(args.input_ann, doc), "r") as in_f:
                doc = doc.replace(".ann", "")
                text = open(os.path.join(args.input_text, "%s.txt" % doc)).read()
                for line in in_f:
                    line = line.strip()
                    if line[0] != "T": continue
                    sp = line.strip().split("\t")
                    assert len(sp) == 3
                    mention = sp[2]
                    sp = sp[1].split(" ")
                    label = sp[0]
                    indices = extract_indices_from_brat_annotation(" ".join(sp[1:]))
                    mention_from_text = _get_mention_from_text(text, indices)
                    if mention != mention_from_text:
                        if sorted(mention.split()) != sorted(mention_from_text.split()):
                            print("Update the mention in document %s from (%s) to (%s)." % (
                            doc, mention, mention_from_text))
                        mention = mention_from_text
                    num_mentions += 1
                    out_f.write("%s\t%s\t%s\t%s\n" % (doc, label, ",".join([str(i) for i in indices]), mention))

    print("Extract %d annotations." % num_mentions)
