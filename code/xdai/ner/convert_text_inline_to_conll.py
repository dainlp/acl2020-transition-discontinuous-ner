'''Usage:
python convert_text_inline_to_conll.py --input_filepath /data/dai031/Experiments/CADEC/adr/flat/train.txt --output_filepath /data/dai031/Experiments/CADEC/adr/flat/train.conll
python convert_text_inline_to_conll.py --input_filepath /data/dai031/Experiments/CADEC/adr/flat/dev.txt --output_filepath /data/dai031/Experiments/CADEC/adr/flat/dev.conll
python convert_text_inline_to_conll.py --input_filepath /data/dai031/Experiments/CADEC/adr/flat/test.txt --output_filepath /data/dai031/Experiments/CADEC/adr/flat/test.conll
'''
import argparse, os, sys

sys.path.insert(0, os.path.abspath("../.."))
from xdai.ner.mention import mentions_to_bio_tags


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    ## Required
    parser.add_argument("--input_filepath", type=str)
    parser.add_argument("--output_filepath", type=str)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_parameters()
    with open(args.output_filepath, "w") as out_f:
        with open(args.input_filepath) as in_f:
            for text in in_f:
                tokens = text.strip().split()
                mentions = next(in_f).strip()
                assert len(next(in_f).strip()) == 0
                tags = mentions_to_bio_tags(mentions.strip(), len(tokens))
                for token, tag in zip(tokens, tags):
                    out_f.write("%s %s\n" % (token, tag))
                out_f.write("\n")