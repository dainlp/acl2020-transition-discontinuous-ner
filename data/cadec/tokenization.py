'''Update date: 2020-Jan-13'''
import argparse, os, sys, re

class Token(object):
    def __init__(self, text=None, start=None, end=None, orig_text=None, text_id=None):
        self.text = text                                            # might be normalized
        self.start = int(start) if start is not None else None      # the character offset of this token into the tokenized sentence.
        if end is not None:
            self.end = int(end)
        else:
            if self.text is not None and self.start is not None:
                self.end = self.start + len(self.text)
            else:
                self.end = None
        self.orig_text = orig_text
        self.text_id = text_id


    def __str__(self):
        return self.text


    def __repr__(self):
        return self.__str__()

class LettersDigitsTokenizer:
    def tokenize(self, text):
        tokens = [Token(m.group(), start=m.start()) for m in re.finditer(r"[^\W\d_]+|\d+|\S", text)]
        return tokens


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    ## Required
    parser.add_argument("--input_dir", default="/data/dai031/Corpora/CADEC/cadec/text", type=str)
    parser.add_argument("--output_filepath", default="/data/dai031/Experiments/CADEC/tokens", type=str)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_parameters()

    tokenizer = LettersDigitsTokenizer()
    num_docs, num_sents, num_tokens = 0, 0, 0

    with open(args.output_filepath, "w") as out_f:
        for doc in os.listdir(args.input_dir):
            with open(os.path.join(args.input_dir, doc), "r") as in_f:
                doc = doc.replace(".txt", "")
                num_docs += 1
                text = in_f.read()
                line_start = 0
                for line in text.split("\n"):
                    line = line.strip()
                    if len(line) > 0:
                        num_sents += 1
                        line_start = text.find(line, line_start)
                        assert line_start >= 0
                        for token in tokenizer.tokenize(line):
                            num_tokens += 1
                            token_start = token.start + line_start
                            token_end = token.end + line_start
                            out_f.write("%s %s %d %d\n" % (token.text, doc, token_start, token_end))
                        out_f.write("\n")
                        line_start += len(line)

    print("%d documents, %d sentences, %s tokens" % (num_docs, num_sents, num_tokens))
