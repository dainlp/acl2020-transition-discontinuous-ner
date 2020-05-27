import torch
from collections import defaultdict
from typing import Dict, List
from xdai.utils.token import Token


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/fields/field.py
Update date: 2019-Nov-5'''
class _Field:
    def count_vocab_items(self, counter):
        pass


    def index(self, vocab):
        pass


    def get_padding_lengths(self):
        raise NotImplementedError


    def as_tensor(self, padding_lengths: Dict[str, int]):
        raise NotImplementedError


    def batch_tensors(self, tensor_list):
        return torch.stack(tensor_list)


    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/fields/metadata_field.py
Update date: 2019-Nov-5'''
class MetadataField(_Field):
    def __init__(self, metadata):
        self.metadata = metadata


    def __getitem__(self, key):
        try:
            return self.metadata[key]
        except TypeError:
            raise TypeError("Metadata is not a dict.")


    def __iter__(self):
        try:
            return iter(self.metadata)
        except TypeError:
            raise TypeError("Metadata is not iterable.")


    def __len__(self):
        try:
            return len(self.metadata)
        except TypeError:
            raise TypeError("Metadata has no length.")


    def get_padding_lengths(self):
        return {}


    def as_tensor(self, padding_lengths):
        return self.metadata


    def empty_field(self):
        return MetadataField(None)


    @classmethod
    def batch_tensors(cls, tensor_list):
        return tensor_list


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#batch_tensor_dicts
Update date: 2019-Nov-5'''
def _batch_tensor_dicts(tensor_dicts):
    '''takes a list of tensor dictionaries, returns a single dictionary with all tensors with the same key batched'''
    key_to_tensors = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)

    batched_tensors = {}
    for key, tensor_list in key_to_tensors.items():
        batched_tensor = torch.stack(tensor_list)
        batched_tensors[key] = batched_tensor
    return batched_tensors


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/fields/text_field.py
Update date: 2019-Nov-5'''
class TextField(_Field):
    def __init__(self, tokens: List[Token], token_indexers):
        self.tokens = tokens
        self._token_indexers = token_indexers
        self._indexed_tokens = None
        self._indexer_name_to_indexed_token = None
        self._token_index_to_indexer_name = None


    def __iter__(self):
        return iter(self.tokens)


    def __getitem__(self, idx):
        return self.tokens[idx]


    def __len__(self):
        return len(self.tokens)


    def count_vocab_items(self, counter):
        for indexer in self._token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)


    def index(self, vocab):
        token_arrays = {}
        indexer_name_to_indexed_token = {}
        token_index_to_indexer_name = {}
        for indexer_name, indexer in self._token_indexers.items():
            token_indices = indexer.tokens_to_indices(self.tokens, vocab, indexer_name)
            token_arrays.update(token_indices)
            indexer_name_to_indexed_token[indexer_name] = list(token_indices.keys())
            for token_index in token_indices:
                token_index_to_indexer_name[token_index] = indexer_name
        self._indexed_tokens = token_arrays
        self._indexer_name_to_indexed_token = indexer_name_to_indexed_token
        self._token_index_to_indexer_name = token_index_to_indexer_name


    def get_padding_lengths(self):
        lengths = []
        assert self._indexed_tokens is not None, "Call .index(vocabulary) before determining padding lengths."
        for indexer_name, indexer in self._token_indexers.items():
            indexer_lengths = {}
            for indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]:
                token_lengths = [indexer.get_padding_lengths(token) for token in self._indexed_tokens[indexed_tokens_key]]
                if not token_lengths:
                    token_lengths = [indexer.get_padding_lengths([])]
                for key in token_lengths[0]:
                    indexer_lengths[key] = max(x[key] if key in x else 0 for x in token_lengths)
            lengths.append(indexer_lengths)

        padding_lengths = {}
        num_tokens = set()
        for token_index, token_list in self._indexed_tokens.items():
            indexer_name = self._token_index_to_indexer_name[token_index]
            indexer = self._token_indexers[indexer_name]
            padding_lengths[f"{token_index}_length"] = max(len(token_list), indexer.get_token_min_padding_length())
            num_tokens.add(len(token_list))
        padding_lengths["num_tokens"] = max(num_tokens)

        padding_keys = {key for d in lengths for key in d.keys()}
        for padding_key in padding_keys:
            padding_lengths[padding_key] = max(x[padding_key] if padding_key in x else 0 for x in lengths)
        return padding_lengths


    def sequence_length(self):
        return len(self.tokens)


    def as_tensor(self, padding_lengths):
        tensors = {}
        for indexer_name, indexer in self._token_indexers.items():
            desired_num_tokens = {indexed_tokens_key: padding_lengths[f"{indexed_tokens_key}_length"] for
                                  indexed_tokens_key in self._indexer_name_to_indexed_token[indexer_name]}
            indices_to_pad = {indexed_tokens_key: self._indexed_tokens[indexed_tokens_key] for indexed_tokens_key in
                              self._indexer_name_to_indexed_token[indexer_name]}
            padded_array = indexer.pad_token_sequence(indices_to_pad, desired_num_tokens, padding_lengths)
            indexer_tensors = {key: torch.LongTensor(array) for key, array in padded_array.items()}
            tensors.update(indexer_tensors)
        return tensors


    def empty_field(self):
        text_field = TextField([], self._token_indexers)
        text_field._indexed_tokens = {}
        text_field._indexer_name_to_indexed_token = {}
        for indexer_name, indexer in self._token_indexers.items():
            array_keys = indexer.get_keys(indexer_name)
            for key in array_keys:
                text_field._indexed_tokens[key] = []
            text_field._indexer_name_to_indexed_token[indexer_name] = array_keys
        return text_field


    def batch_tensors(self, tensor_dicts):
        return _batch_tensor_dicts(tensor_dicts)


    '''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#get_text_field_mask'''
    @classmethod
    def get_text_field_mask(cls, text_field_tensors: Dict[str, torch.Tensor]):
        if "mask" in text_field_tensors:
            return text_field_tensors["mask"]

        tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
        tensor_dims.sort(key=lambda x: x[0])

        assert tensor_dims[0][0] == 2

        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/instance.py
An instance is a collection of Field objects, specifying the inputs and outputs to the model.
Update date: 2019-Nov-5'''
class Instance:
    def __init__(self, fields):
        self.fields = fields
        self.indexed = False


    def __getitem__(self, key):
        return self.fields[key]


    def __iter__(self):
        return iter(self.fields)


    def __len__(self):
        return len(self.fields)


    def add_field(self, field_name, field, vocab):
        self.fields[field_name] = field
        if self.indexed:
            field.index(vocab)


    def count_vocab_items(self, counter):
        for field in self.fields.values():
            field.count_vocab_items(counter)


    def index_fields(self, vocab):
        if not self.indexed:
            self.indexed = True
            for field in self.fields.values():
                field.index(vocab)


    def get_padding_lengths(self):
        lengths = {}
        for field_name, field in self.fields.items():
            lengths[field_name] = field.get_padding_lengths()
        return lengths


    def as_tensor_dict(self, padding_lengths):
        padding_lengths = padding_lengths or self.get_padding_lengths()
        tensors = {}
        for field_name, field in self.fields.items():
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
        return tensors
