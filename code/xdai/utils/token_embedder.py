import inspect, os, torch
import numpy as np
from typing import Dict
from xdai.utils.nn import TimeDistributed
from xdai.utils.seq2vec import CnnEncoder
from xdai.elmo.models import Elmo


'''Update date: 2019-Nov-5'''
class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, weight: torch.FloatTensor = None, trainable=True):
        super(Embedding, self).__init__()
        self.output_dim = embedding_dim

        if weight is None:
            weight = torch.FloatTensor(vocab_size, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            assert weight.size() == (vocab_size, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)


    def get_output_dim(self):
        return self.output_dim


    def forward(self, inputs):
        outs = torch.nn.functional.embedding(inputs, self.weight)
        return outs


'''Update date: 2019-Nov-5'''
class TokenCharactersEmbedder(torch.nn.Module):
    def __init__(self, embedding: Embedding, encoder, dropout=0.0):
        super(TokenCharactersEmbedder, self).__init__()
        self._embedding = TimeDistributed(embedding)
        self._encoder = TimeDistributed(encoder)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x


    def get_output_dim(self):
        return self._encoder._module.get_output_dim()


    def forward(self, token_characters):
        '''token_characters: batch_size, num_tokens, num_characters'''
        mask = (token_characters != 0).long()
        outs = self._embedding(token_characters)
        outs = self._encoder(outs, mask)
        outs = self._dropout(outs)
        return outs


'''A single layer of ELMo representations, essentially a wrapper around ELMo(num_output_representations=1, ...)
Update date: 2019-Nov-5'''
class ElmoTokenEmbedder(torch.nn.Module):
    def __init__(self, options_file, weight_file, dropout=0.5, requires_grad=False,
                 projection_dim=None):
        super(ElmoTokenEmbedder, self).__init__()

        self._elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=dropout,
                          requires_grad=requires_grad)
        if projection_dim:
            self._projection = torch.nn.Linear(self._elmo.get_output_dim(), projection_dim)
            self.output_dim = projection_dim
        else:
            self._projection = None
            self.output_dim = self._elmo.get_output_dim()


    def get_output_dim(self):
        return self.output_dim


    def forward(self, inputs):
        # inputs: batch_size, num_tokens, 50
        elmo_output = self._elmo(inputs)
        elmo_representations = elmo_output["elmo_representations"][0]
        if self._projection:
            projection = self._projection
            for _ in range(elmo_representations.dim() - 2):
                projection = TimeDistributed(projection)
            elmo_representations = projection(elmo_representations)
        return elmo_representations


'''Update date: 2019-Nov-5'''
def _load_pretrained_embeddings(filepath, dimension, token2idx):
    tokens_to_keep = set(token2idx.keys())
    embeddings = {}
    if filepath != "" and os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                sp = line.strip().split(" ")
                if len(sp) <= dimension: continue
                token = sp[0]
                if token not in tokens_to_keep: continue
                embeddings[token] = np.array([float(x) for x in sp[1:]])

    print(" # Load %d out of %d words (%d-dimensional) from pretrained embedding file (%s)!" % (
    len(embeddings), len(token2idx), dimension, filepath))

    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    weights = np.random.normal(embeddings_mean, embeddings_std, size=(len(token2idx), dimension))
    for token, i in token2idx.items():
        if token in embeddings:
            weights[i] = embeddings[token]
    return weights


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/text_field_embedders/*
Takes as input the dict produced by TextField and 
returns as output an embedded representations of the tokens in that field
Update date: 2019-Nov-5'''
class TextFieldEmbedder(torch.nn.Module):
    def __init__(self, token_embedders, embedder_to_indexer_map=None):
        super(TextFieldEmbedder, self).__init__()

        self.token_embedders = token_embedders
        self._embedder_to_indexer_map = embedder_to_indexer_map
        for k, embedder in token_embedders.items():
            self.add_module("token_embedder_%s" % k, embedder)


    def get_output_dim(self):
        return sum([embedder.get_output_dim() for embedder in self.token_embedders.values()])


    '''text_field_input is the output of a call to TextField.as_tensor (see instance.py).
    Each tensor in here is assumed to have a shape roughly similar to (batch_size, num_tokens)'''
    def forward(self, text_field_input: Dict[str, torch.Tensor], **kwargs):
        outs = []
        for k in sorted(self.token_embedders.keys()):
            embedder = getattr(self, "token_embedder_%s" % k)
            forward_params = inspect.signature(embedder.forward).parameters
            forward_params_values = {}
            for param in forward_params.keys():
                if param in kwargs:
                    forward_params_values[param] = kwargs[param]
            if self._embedder_to_indexer_map is not None and k in self._embedder_to_indexer_map:
                indexer_map = self._embedder_to_indexer_map[k]
                assert isinstance(indexer_map, dict)
                tensors = {name: text_field_input[argument] for name, argument in indexer_map.items()}
                outs.append(embedder(**tensors, **forward_params_values))
            else:
                tensors = [text_field_input[k]]
                outs.append(embedder(*tensors, **forward_params_values))
        return torch.cat(outs, dim=-1)


    @classmethod
    def tokens_embedder(cls, vocab, args):
        token2idx = vocab.get_item_to_index_vocabulary("tokens")
        weight = _load_pretrained_embeddings(args.pretrained_word_embeddings, dimension=100, token2idx=token2idx)
        return Embedding(len(token2idx), embedding_dim=100, weight=torch.FloatTensor(weight))


    @classmethod
    def token_characters_embedder(cls, vocab, args):
        embedding = Embedding(vocab.get_vocab_size("token_characters"), embedding_dim=16)
        return TokenCharactersEmbedder(embedding, CnnEncoder())


    @classmethod
    def elmo_embedder(cls, vocab, args):
        option_file = os.path.join(args.pretrained_model_dir, "options.json")
        weight_file = os.path.join(args.pretrained_model_dir, "weights.hdf5")
        return ElmoTokenEmbedder(option_file, weight_file)


    @classmethod
    def create_embedder(cls, args, vocab):
        embedder_to_indexer_map = {}
        embedders = {"tokens": TextFieldEmbedder.tokens_embedder(vocab, args),
                 "token_characters": TextFieldEmbedder.token_characters_embedder(vocab, args)}

        if args.model_type == "elmo":
            embedders["elmo_characters"] = TextFieldEmbedder.elmo_embedder(vocab, args)
        return cls(embedders, embedder_to_indexer_map)