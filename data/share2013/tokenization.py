import argparse, logging, os, sys
from typing import List, NamedTuple

logger = logging.getLogger(__name__)


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default=None, type=str)
    parser.add_argument("--split", default=None, type=str)
    parser.add_argument("--log_filepath", default="output.log", type=str)

    args, _ = parser.parse_known_args()
    return args


class Token(NamedTuple):
    text: str = None
    idx: int = None     # start character offset


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/word_splitter.py#SimpleWordSplitter'''
class CustomSplitter:
    def __init__(self):
        self.special_cases = set(["mr.", "mrs.", "etc.", "e.g.", "cf.", "c.f.", "eg.", "al."])
        self.special_beginning = set(["http", "www"])
        self.contractions = set(["n't", "'s", "'ve", "'re", "'ll", "'d", "'m"])
        self.contractions |= set([x.replace("'", "’") for x in self.contractions])
        self.ending_punctuation = set(['"', "'", '.', ',', ';', ')', ']', '}', ':', '!', '?', '%', '”', "’"])
        self.beginning_punctuation = set(['"', "'", '(', '[', '{', '#', '$', '“', "‘", "+", "*", "="])
        self.delimiters = set(["-", "/", ",", ")", "&", "(", "?", ".", "\\", ";", ":", "+", ">", "%"])

    def split_tokens(self, sentence: str) -> List[Token]:
        original_sentence = sentence
        sentence = list(sentence)
        sentence = "".join([o if not o in self.delimiters else " %s " % o for o in sentence])
        tokens = []
        field_start, field_end = 0, 0
        for filed in sentence.split():
            filed = filed.strip()
            if len(filed) == 0: continue

            field_start = original_sentence.find(filed, field_start)
            field_end = field_start + len(filed)
            assert field_start >= 0, "cannot find (%s) from \"%s\" after offset %d" % (
            filed, original_sentence, field_start)

            add_at_end = []
            while self._can_split(filed) and filed[0] in self.beginning_punctuation:
                tokens.append(Token(filed[0], field_start))
                filed = filed[1:]
                field_start += 1

            while self._can_split(filed) and filed[-1] in self.ending_punctuation:
                add_at_end.insert(0, Token(filed[-1], field_start + len(filed) - 1))
                filed = filed[:-1]

            remove_contractions = True
            while remove_contractions:
                remove_contractions = False
                for contraction in self.contractions:
                    if self._can_split(filed) and filed.lower().endswith(contraction):
                        add_at_end.insert(0, Token(filed[-len(contraction):], field_start + len(filed) - len(contraction)))
                        filed = filed[:-len(contraction)]
                        remove_contractions = True

            if filed:
                tokens.append(Token(filed, field_start))
            tokens.extend(add_at_end)
            field_start = field_end
        return tokens

    def _can_split(self, token):
        if not token: return False
        if token.lower() in self.special_cases: return False
        for _special_beginning in self.special_beginning:
            if token.lower().startswith(_special_beginning): return False
        return True


class CustomSentenceSplitter():
    def _next_character_is_upper(self, text, i):
        while i < len(text):
            if len(text[i].strip()) == 0:
                i += 1
            elif text[i].isupper():
                return True
            else:
                break
        return False

    # do very simple things: if there is a period '.', and the next character is uppercased, call it a sentence.
    def split_sentence(self, text):
        break_points = [0]
        for i in range(len(text)):
            if text[i] in [".", "!", "?"]:
                if self._next_character_is_upper(text, i + 1):
                    break_points.append(i + 1)
        break_points.append(-1)
        sentences = []
        for s, e in zip(break_points[0:-1], break_points[1:]):
            if e == -1:
                sentences.append(text[s:].strip())
            else:
                sentences.append(text[s:e].strip())
        return sentences


if __name__ == "__main__":
    args = parse_parameters()
    handlers = [logging.FileHandler(filename=args.log_filepath), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, handlers=handlers)

    token_splitter = CustomSplitter()
    sentence_splitter = CustomSentenceSplitter()

    with open("%s.tokens" % args.split, "w") as out_f:
        num_of_sentences, num_tokens = 0, 0
        for filename in os.listdir(args.input_dir):
            if not filename.endswith("txt"): continue
            with open(os.path.join(args.input_dir, filename)) as in_f:
                text = in_f.read()
                token_start, token_end = 0, 0
                for i, line in enumerate(text.splitlines()):
                    if len(line.strip()) > 0:
                        if i <= 5 or len(line.strip()) < 150:
                            num_of_sentences += 1
                            tokens = token_splitter.split_tokens(line.strip())
                            for token in tokens:
                                num_tokens += 1
                                token_start = text.find(token.text, token_start)
                                assert token_start >= 0
                                token_end = token_start + len(token.text.strip())
                                out_f.write("%s %s %d %d\n" % (token.text, filename, token_start, token_end))
                                token_start = token_end
                            out_f.write("\n")
                        else:
                            for sentence in sentence_splitter.split_sentence(line.strip()):
                                num_of_sentences += 1
                                tokens = token_splitter.split_tokens(sentence.strip())
                                for token in tokens:
                                    num_tokens += 1
                                    token_start = text.find(token.text, token_start)
                                    assert token_start >= 0
                                    token_end = token_start + len(token.text.strip())
                                    out_f.write("%s %s %d %d\n" % (token.text, filename, token_start, token_end))
                                    token_start = token_end
                                out_f.write("\n")

        logger.info(f"{num_of_sentences} sentences and {num_tokens} tokens in {args.input_dir}")