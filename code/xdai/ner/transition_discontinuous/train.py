import json, logging, os, sys, torch
sys.path.insert(0, os.path.abspath("../../.."))

from xdai.utils.args import parse_parameters
from xdai.utils.common import create_output_dir, pad_sequence_to_length, set_cuda, set_random_seed
from xdai.utils.instance import Instance, MetadataField, TextField
from xdai.utils.iterator import BasicIterator, BucketIterator
from xdai.utils.token import Token
from xdai.utils.token_indexer import SingleIdTokenIndexer, TokenCharactersIndexer, ELMoIndexer
from xdai.utils.train import train_op, eval_op
from xdai.utils.vocab import Vocabulary
from xdai.ner.transition_discontinuous.models import TransitionModel
from xdai.ner.mention import Mention
from xdai.ner.transition_discontinuous.parsing import Parser


logger = logging.getLogger(__name__)


'''Update at April-22-2019'''
class ActionField:
    def __init__(self, actions, inputs):
        self._key = "actions"
        self.actions = actions
        self._indexed_actions = None
        self.inputs = inputs

        if all([isinstance(a, int) for a in actions]):
            self._indexed_actions = actions


    def count_vocab_items(self, counter):
        if self._indexed_actions is None:
            for action in self.actions:
                counter[self._key][action] += 1


    def index(self, vocab):
        if self._indexed_actions is None:
            self._indexed_actions = [vocab.get_item_index(action, self._key) for action in self.actions]


    def get_padding_lengths(self):
        return {"num_tokens": self.inputs.sequence_length() * 2}


    def as_tensor(self, padding_lengths):
        desired_num_actions = padding_lengths["num_tokens"]
        padded_actions = pad_sequence_to_length(self._indexed_actions, desired_num_actions)
        return torch.LongTensor(padded_actions)


    def batch_tensors(self, tensor_list):
        return torch.stack(tensor_list)


class DatasetReader:
    def __init__(self, args):
        self.args = args
        self.parse = Parser()
        self._token_indexers = {"tokens": SingleIdTokenIndexer(), "token_characters": TokenCharactersIndexer()}
        if args.model_type == "elmo":
            self._token_indexers["elmo_characters"] = ELMoIndexer()


    def read(self, filepath, training=False):
        instances = []
        with open(filepath, "r") as f:
            for sentence in f:
                tokens = [Token(t) for t in sentence.strip().split()]
                annotations = next(f).strip()
                actions = self.parse.mention2actions(annotations, len(tokens))
                oracle_mentions = [str(s) for s in self.parse.parse(actions, len(tokens))]
                gold_mentions = annotations.split("|") if len(annotations) > 0 else []

                if len(oracle_mentions) != len(gold_mentions) or len(oracle_mentions) != len(
                        set(oracle_mentions) & set(gold_mentions)):
                    logger.debug("Discard this instance whose oracle mention is: %s, while its gold mention is: %s" % (
                    "|".join(oracle_mentions), annotations))
                    if not training:
                        instances.append(self._to_instance(sentence, annotations, tokens, actions))
                else:
                    instances.append(self._to_instance(sentence, annotations, tokens, actions))

                assert len(next(f).strip()) == 0
        return instances


    def _to_instance(self, sentence, annotations, tokens, actions):
        text_fields = TextField(tokens, self._token_indexers)
        action_fields = ActionField(actions, text_fields)
        sentence = MetadataField(sentence.strip())
        annotations = MetadataField(annotations.strip())
        return Instance(
            {"sentence": sentence, "annotations": annotations, "tokens": text_fields, "actions": action_fields})


if __name__ == "__main__":
    args = parse_parameters()
    create_output_dir(args)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, filename=args.log_filepath)
    addition_args = json.load(open("config.json"))
    for k, v in addition_args.items():
        setattr(args, k, v)
    logger.info(
        "Parameters: %s" % json.dumps({k: v for k, v in vars(args).items() if v is not None}, indent=2, sort_keys=True))

    set_cuda(args)
    set_random_seed(args)

    dataset_reader = DatasetReader(args)
    train_data = dataset_reader.read(args.train_filepath, training=True)
    if args.dev_filepath is None:
        num_dev_instances = int(len(train_data) / 10)
        dev_data = train_data[0:num_dev_instances]
        train_data = train_data[num_dev_instances:]
    else:
        dev_data = dataset_reader.read(args.dev_filepath)
    if args.num_train_instances is not None: train_data = train_data[0:args.num_train_instances]
    if args.num_dev_instances is not None: dev_data = dev_data[0:args.num_dev_instances]
    logger.info("Load %d instances from train set." % (len(train_data)))
    logger.info("Load %d instances from dev set." % (len(dev_data)))
    test_data = dataset_reader.read(args.test_filepath)
    logger.info("Load %d instances from test set." % (len(test_data)))

    datasets = {"train": train_data, "validation": dev_data, "test": test_data}
    vocab = Vocabulary.from_instances((instance for dataset in datasets.values() for instance in dataset))
    vocab.save_to_files(os.path.join(args.output_dir, "vocabulary"))
    train_iterator = BucketIterator(sorting_keys=[['tokens', 'tokens_length']], batch_size=args.train_batch_size_per_gpu)
    train_iterator.index_with(vocab)
    dev_iterator = BasicIterator(batch_size=args.eval_batch_size_per_gpu)
    dev_iterator.index_with(vocab)

    model = TransitionModel(args, vocab).cuda(args.cuda_device[0])
    parameters = [p for _, p in model.named_parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    metrics = train_op(args, model, optimizer, train_data, train_iterator, dev_data, dev_iterator)
    logger.info(metrics)

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best.th")))
    test_metrics, test_preds = eval_op(args, model, test_data, dev_iterator)
    logger.info(test_metrics)
    with open(os.path.join(args.output_dir, "test.pred"), "w") as f:
        for i in test_preds:
            f.write("%s\n%s\n\n" % (i[0], i[1]))

    if args.dev_filepath is not None:
        dev_metrics, dev_preds = eval_op(args, model, dev_data, dev_iterator)
        logger.info(dev_metrics)
        with open(os.path.join(args.output_dir, "dev.pred"), "w") as f:
            for i in dev_preds:
                f.write("%s\n%s\n\n" % (i[0], i[1]))