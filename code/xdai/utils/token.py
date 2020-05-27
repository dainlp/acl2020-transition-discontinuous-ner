import re, spacy
from typing import List, NamedTuple
from xdai.utils.common import load_spacy_model


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/token.py
Update date: 2019-Nov-5'''
class Token(NamedTuple):
    text: str = None
    start: int = None  # the character offset of this token into the tokenized sentence.


    @property
    def end(self):
        return self.start + len(self.text)


    def __str__(self):
        return self.text


    def __repr__(self):
        return self.__str__()


'''Update date: 2019-Nov-26'''
def preprocess_twitter(text):
    tokens = []
    for token in text.strip().split():
        if len(token) > 1 and (token[0] == "@" or token[0] == "#"):
            tokens.append(token[0])
            tokens.append(token[1:])
        else:
            tokens.append(token)
    return " ".join(tokens)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/letters_digits_tokenizer.py
Update date: 2019-Nov-25'''
class LettersDigitsTokenizer:
    def tokenize(self, text):
        tokens = [Token(m.group(), start=m.start()) for m in re.finditer(r"[^\W\d_]+|\d+|\S", text)]
        return tokens


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/whitespace_tokenizer.py
Update date: 2019-Nov-25'''
class WhitespaceTokenizer:
    def tokenize(self, text):
        return [Token(t) for t in text.split()]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/spacy_tokenizer.py#_remove_spaces
Update date: 2019-Nov-25'''
def _remove_spaces(tokens: List[spacy.tokens.Token]):
    return [t for t in tokens if not t.is_space]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/spacy_tokenizer.py
Update date: 2019-Nov-25'''
class SpacyTokenizer:
    def __init__(self, language="en_core_web_sm"):
        self.spacy = load_spacy_model(language)


    def _sanitize(self, tokens):
        return [Token(t.text, t.idx) for t in tokens]


    def batch_tokenize(self, texts: List[str]):
        return [self._sanitize(_remove_spaces(tokens)) for tokens in self.spacy.pipe(texts, n_threads=-1)]


    def tokenize(self, text):
        return self._sanitize(_remove_spaces(self.spacy(text)))


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/tokenizers/sentence_splitter.py
Update date: 2019-Nov-25'''
class SpacySentenceSplitter:
    def __init__(self, language="en_core_web_sm", rule_based=False):
        self.spacy = load_spacy_model(language, parse=not rule_based)
        if rule_based:
            sbd_name = "sbd" if spacy.__version__ < "2.1" else "sentencizer"
            if not self.spacy.has_pipe(sbd_name):
                sbd = self.spacy.create_pipe(sbd_name)
                self.spacy.add_pipe(sbd)


    def split_sentences(self, text: str) -> List[str]:
        return [sent.string.strip() for sent in self.spacy(text).sents]


    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        return [[sent.string.strip() for sent in doc.sents] for doc in self.spacy.pipe(texts)]