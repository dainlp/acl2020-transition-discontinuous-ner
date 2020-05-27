'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/data/vocabulary.py
Update date: 2019-Nov-5'''
import codecs, logging, os
from collections import defaultdict
from typing import Dict
from tqdm import tqdm


logger = logging.getLogger(__name__)


DEFAULT_NON_PADDED_NAMESPACES = ["tags", "labels"]


class _NamespaceDependentDefaultDict(defaultdict):
    def __init__(self, padded_function, non_padded_function):
        # we do not take non_padded_namespaces as a parameter,
        # because we consider any namespace whose key name ends with labels or tags as non padded namespace,
        # and use padded namespace otherwise
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_NamespaceDependentDefaultDict, self).__init__()


    def __missing__(self, key):
        if any(key.endswith(pattern) for pattern in DEFAULT_NON_PADDED_NAMESPACES):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value


class _ItemToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, padding_item, oov_item):
        super(_ItemToIndexDefaultDict, self).__init__(lambda: {padding_item: 0, oov_item: 1},
                                                       lambda: {})


class _IndexToItemDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, padding_item, oov_item):
        super(_IndexToItemDefaultDict, self).__init__(lambda: {0: padding_item, 1: oov_item},
                                                       lambda: {})


class Vocabulary:
    def __init__(self, counter: Dict[str, Dict[str, int]] = None, min_count: Dict[str, int] = None,
                 max_vocab_size: Dict[str, int] = None):
        self._padding_item = "@@PADDING@@"
        self._oov_item = "@@UNKNOWN@@"
        self._item_to_index = _ItemToIndexDefaultDict(self._padding_item, self._oov_item)
        self._index_to_item = _IndexToItemDefaultDict(self._padding_item, self._oov_item)
        self._extend(counter=counter, min_count=min_count, max_vocab_size=max_vocab_size)


    '''Update date: 2019-Nov-9'''
    def save_to_files(self, directory):
        os.makedirs(directory, exist_ok=True)
        for namespace, mapping in self._index_to_item.items():
            with codecs.open(os.path.join(directory, "%s.txt" % namespace), "w", "utf-8") as f:
                for i in range(len(mapping)):
                    f.write("%s\n" % (mapping[i].replace("\n", "@@NEWLINE@@").strip()))


    '''Update date: 2019-Nov-9'''
    @classmethod
    def from_files(cls, directory):
        logger.info("Loading item dictionaries from %s.", directory)
        vocab = cls()
        for namespace in os.listdir(directory):
            if not namespace.endswith(".txt"): continue
            with codecs.open(os.path.join(directory, namespace), "r", "utf-8") as f:
                namespace = namespace.replace(".txt", "")
                for i, line in enumerate(f):
                    line = line.strip()
                    if len(line) == 0: continue
                    item = line.replace("@@NEWLINE@@", "\n")
                    vocab._item_to_index[namespace][item] = i
                    vocab._index_to_item[namespace][i] = item
        return vocab


    @classmethod
    def from_instances(cls, instances, min_count=None, max_vocab_size=None):
        counter = defaultdict(lambda: defaultdict(int))
        for instance in tqdm(instances):
            instance.count_vocab_items(counter)
        return cls(counter=counter, min_count=min_count, max_vocab_size=max_vocab_size)


    def _extend(self, counter, min_count=None, max_vocab_size=None):
        counter = counter or {}
        min_count = min_count or {}
        max_vocab_size = max_vocab_size or {}
        for namespace in counter:
            item_counts = list(counter[namespace].items())
            item_counts.sort(key=lambda x: x[1], reverse=True)

            if namespace in max_vocab_size and max_vocab_size[namespace] > 0:
                item_counts = item_counts[:max_vocab_size[namespace]]
            for item, count in item_counts:
                if count >= min_count.get(namespace, 1):
                    self._add_item_to_namespace(item, namespace)


    def _add_item_to_namespace(self, item, namespace="tokens"):
        if item not in self._item_to_index[namespace]:
            idx = len(self._item_to_index[namespace])
            self._item_to_index[namespace][item] = idx
            self._index_to_item[namespace][idx] = item


    def get_index_to_item_vocabulary(self, namespace="tokens"):
        return self._index_to_item[namespace]


    def get_item_to_index_vocabulary(self, namespace="tokens"):
        return self._item_to_index[namespace]


    def get_item_index(self, item, namespace="tokens"):
        if item in self._item_to_index[namespace]:
            return self._item_to_index[namespace][item]
        else:
            return self._item_to_index[namespace][self._oov_item]


    def get_item_from_index(self, idx: int, namespace="tokens"):
        return self._index_to_item[namespace][idx]


    def get_vocab_size(self, namespace="tokens"):
        return len(self._item_to_index[namespace])