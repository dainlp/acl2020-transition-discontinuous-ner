import json, logging, os, random, shutil, spacy, torch
import numpy as np
from typing import Any, Dict


logger = logging.getLogger(__name__)


'''Update date: 2019-Nov-3'''
def create_output_dir(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        if args.overwrite_output_dir:
            shutil.rmtree(args.output_dir)
        else:
            raise ValueError("Output directory (%s) already exists." % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py#dump_metrics
Update date: 2019-Nov-3'''
def dump_metrics(file_path: str, metrics: Dict[str, Any], log: bool = False) -> None:
    metrics_json = json.dumps(metrics, indent=2)
    with open(file_path, "w") as metrics_file:
        metrics_file.write(metrics_json)
    if log:
        logger.info("Metrics: %s", metrics_json)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-April-26'''
def has_tensor(obj) -> bool:
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, dict):
        return any(has_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(has_tensor(i) for i in obj)
    return False


LOADED_SPACY_MODELS = {}
'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py#get_spacy_model
Update date: 2019-Nov-25'''
def load_spacy_model(spacy_model_name, parse=False):
    options = (spacy_model_name, parse)
    if options not in LOADED_SPACY_MODELS:
        disable = ["vectors", "textcat", "tagger", "ner"]
        if not parse:
            disable.append("parser")
        spacy_model = spacy.load(spacy_model_name, disable=disable)
        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/metric.py
Update date: 2019-03-01'''
def move_to_cpu(*tensors):
    return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py (move_to_device)
Update date: 2019-April-26'''
def move_to_gpu(obj, cuda_device=0):
    if cuda_device < 0 or not has_tensor(obj): return obj
    if isinstance(obj, torch.Tensor): return obj.cuda(cuda_device)
    if isinstance(obj, dict):
        return {k: move_to_gpu(v, cuda_device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_gpu(v, cuda_device) for v in obj]
    if isinstance(obj, tuple):
        return tuple([move_to_gpu(v, cuda_device) for v in obj])
    return obj


'''Update date: 2019-Nov-3'''
def pad_sequence_to_length(sequence, desired_length, default_value=lambda: 0):
    padded_sequence = sequence[:desired_length]
    for _ in range(desired_length - len(padded_sequence)):
        padded_sequence.append(default_value())
    return padded_sequence


'''Update date: 2019-Nov-4'''
def set_cuda(args):
    cuda_device = [int(i) for i in args.cuda_device.split(",")]
    args.cuda_device = [i for i in cuda_device if i >= 0]
    args.n_gpu = len(args.cuda_device)
    logger.info("Device: %s, n_gpu: %s" % (args.cuda_device, args.n_gpu))


'''Update date: 2019-Nov-3'''
def set_random_seed(args):
    if args.seed <= 0:
        logger.info("Does not set the random seed, since the value is %s" % args.seed)
        return
    random.seed(args.seed)
    np.random.seed(int(args.seed / 2))
    torch.manual_seed(int(args.seed / 4))
    torch.cuda.manual_seed_all(int(args.seed / 8))


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#sort_batch_by_length
Update date: 2019-Nov-5'''
def sort_batch_by_length(tensor, sequence_lengths):
    '''restoration_indices: sorted_tensor.index_select(0, restoration_indices) == original_tensor'''
    assert isinstance(tensor, torch.Tensor) and isinstance(sequence_lengths, torch.Tensor)

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)

    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index