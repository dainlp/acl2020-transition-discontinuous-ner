import itertools, re
from xdai.utils.common import pad_sequence_to_length


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/token_indexer.py#TokenIndexer
Update date: 2019-Nov-5'''
class _TokenIndexer:
    '''A ``TokenIndexer`` determines how string tokens get represented as arrays of indices in a model.'''
    def __init__(self, token_min_padding_length=0):
        self._token_min_padding_length = token_min_padding_length


    def tokens_to_indices(self, tokens, vocabulary, index_name):
        '''Take a list of tokens and convert them to one or more sets of indices.'''
        raise NotImplementedError


    def get_padding_token(self):
        '''When we need to add padding tokens, what should they look like? A 'blank' token.'''
        raise NotImplementedError


    def get_padding_lengths(self, token):
        raise NotImplementedError


    def get_token_min_padding_length(self):
        return self._token_min_padding_length


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/single_id_token_indexer.py
Update date: 2019-Nov-5'''
class SingleIdTokenIndexer(_TokenIndexer):
    def __init__(self, lowercase_tokens=True, normalize_digits=False, token_min_padding_length=0):
        super().__init__(token_min_padding_length)
        self.namespace = "tokens"
        self.lowercase_tokens = lowercase_tokens
        self.normalize_digits = normalize_digits


    def count_vocab_items(self, token, counter):
        text = token.text
        if self.lowercase_tokens: text = text.lower()
        if self.normalize_digits: text = re.sub(r"[0-9]", "0", text)
        counter[self.namespace][text] += 1


    def tokens_to_indices(self, tokens, vocabulary, index_name):
        indices = []
        for token in tokens:
            text = token.text
            if self.lowercase_tokens: text = text.lower()
            if self.normalize_digits: text = re.sub(r"[0-9]", "0", text)
            indices.append(vocabulary.get_item_index(text, self.namespace))
        return {index_name: indices}


    def get_padding_token(self):
        return 0


    def get_padding_lengths(self, token):
        return {}


    '''tokens: {'tokens': [53, 10365, 9, 53, 15185, 10]}
    desired_num_tokens: {'tokens': 11}
    return: {'tokens': [53, 10365, 9, 53, 15185, 10, 0, 0, 0, 0, 0]}'''
    def pad_token_sequence(self, tokens, desired_num_tokens, padding_lengths=None):
        return {k: pad_sequence_to_length(v, desired_num_tokens[k]) for k, v in tokens.items()}


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/token_characters_indexer.py
Update date: 2019-Nov-5'''
class TokenCharactersIndexer(_TokenIndexer):
    def __init__(self, token_min_padding_length=0):
        super().__init__(token_min_padding_length)
        self._namespace = "token_characters"
        # If using CnnEncoder to build character-level representations,
        # this value is set to the maximum value of ngram_filter_sizes
        self._min_padding_length = 3


    def count_vocab_items(self, token, counter):
        for c in list(token.text):
            counter[self._namespace][c] += 1


    def tokens_to_indices(self, tokens, vocabulary, index_name):
        indices = []
        for token in tokens:
            token_indices = []
            for c in list(token.text):
                index = vocabulary.get_item_index(c, self._namespace)
                token_indices.append(index)
            indices.append(token_indices)
        return {index_name: indices}


    def get_padding_lengths(self, token):
        return {"num_token_characters": max(len(token), self._min_padding_length)}


    def get_padding_token(self):
        return []


    '''tokens: {'token_characters': [[45, 8, 6, 4, 6, 9, 12], [52, 3, 4, 3], 
                            [6, 5], [15, 2, 8, 18, 2, 8], [4, 3, 10, 30, 9], [21, 6, 4, 12], 
                            [42, 2, 5, 4, 15, 7, 8, 2], [19]]}
    desired_num_tokens: {'token_characters': 10}
    padding_lengths: {'tokens_length': 10, 'token_characters_length': 10, 'num_tokens': 10, 'num_token_characters': 12}
    return: {'token_characters': [[45, 8, 6, 4, 6, 9, 12, 0, 0, 0, 0, 0], [52, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0], 
                                [6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [15, 2, 8, 18, 2, 8, 0, 0, 0, 0, 0, 0], 
                                [4, 3, 10, 30, 9, 0, 0, 0, 0, 0, 0, 0], [21, 6, 4, 12, 0, 0, 0, 0, 0, 0, 0, 0], 
                                [42, 2, 5, 4, 15, 7, 8, 2, 0, 0, 0, 0], [19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
    '''
    def pad_token_sequence(self, tokens, desired_num_tokens, padding_lengths):
        padded_tokens = pad_sequence_to_length(tokens[self._namespace],
                                               desired_length=desired_num_tokens[self._namespace],
                                               default_value=self.get_padding_token)

        desired_token_length = padding_lengths["num_token_characters"]
        longest_token_length = max([len(t) for t in tokens[self._namespace]])

        if desired_token_length > longest_token_length:
            padded_tokens.append([0] * desired_token_length)

        padded_tokens = list(zip(*itertools.zip_longest(*padded_tokens, fillvalue=0)))
        if desired_token_length > longest_token_length:
            padded_tokens.pop()

        return {self._namespace: [list(token[:desired_token_length]) for token in padded_tokens]}


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/elmo_indexer.py#ELMoCharacterMapper
Update date: 2019-Nov-5'''
class ELMoCharacterMapper:
    max_word_length = 50
    # 0-255 for utf-8 encoding bytes
    beginning_of_sentence_character = 256
    end_of_sentence_character = 257
    beginning_of_word_character = 258
    end_of_word_character = 259
    padding_character = 260

    beginning_of_sentence_characters = [padding_character] * max_word_length
    beginning_of_sentence_characters[0] = beginning_of_word_character
    beginning_of_sentence_characters[1] = beginning_of_sentence_character
    beginning_of_sentence_characters[2] = end_of_word_character

    end_of_sentence_characters = [padding_character] * max_word_length
    end_of_sentence_characters[0] = beginning_of_word_character
    end_of_sentence_characters[1] = end_of_sentence_character
    end_of_sentence_characters[2] = end_of_word_character

    bos_token = "<S>"
    eos_token = "</S>"


    @staticmethod
    def convert_word_to_char_ids(word):
        if word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode("utf-8", "ignore")[:(ELMoCharacterMapper.max_word_length - 2)]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for i, v in enumerate(word_encoded, start=1):
                char_ids[i] = v
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character
        return [c + 1 for c in char_ids] # add 1 for masking


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/elmo_indexer.py#ELMoTokenCharactersIndexer
Update date: 2019-Nov-5'''
class ELMoIndexer(_TokenIndexer):
    def __init__(self, token_min_padding_length=0):
        super().__init__(token_min_padding_length)
        self._namespace = "elmo_characters"


    def count_vocab_items(self, token, counter):
        pass


    def tokens_to_indices(self, tokens, vocabulary, index_name):
        texts = [token.text for token in tokens]
        return {index_name: [ELMoCharacterMapper.convert_word_to_char_ids(text) for text in texts]}


    def get_padding_lengths(self, token):
        return {}


    def get_padding_token(self):
        return []


    @staticmethod
    def _default_value_for_padding():
        return [0] * ELMoCharacterMapper.max_word_length


    def pad_token_sequence(self, tokens, desired_num_tokens, padding_lengths):
        return {k: pad_sequence_to_length(v, desired_length=desired_num_tokens[k],
                                          default_value=self._default_value_for_padding) for k, v in tokens.items()}