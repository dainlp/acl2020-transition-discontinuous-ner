'''Update date: 2020-Jan-13'''
import argparse
from typing import List


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    ## Required
    parser.add_argument("--input_ann", default="/data/dai031/Experiments/CADEC/ann", type=str)
    parser.add_argument("--input_tokens", default="/data/dai031/Experiments/CADEC/tokens", type=str)
    parser.add_argument("--output_ann", default="/data/dai031/Experiments/CADEC/tokens.ann", type=str)

    args, _ = parser.parse_known_args()
    return args


def merge_consecutive_indices(indices: List[int]) -> List[int]:
    '''convert 136 142 143 147 into 136 147 (these two spans are actually consecutive),
    136 142 143 147 148 160 into 136 160 (these three spans are consecutive)
    it only makes sense when these indices are inclusive'''
    consecutive_indices = []
    assert len(indices) % 2 == 0
    for i, v in enumerate(indices):
        if (i == 0) or (i == len(indices) - 1):
            consecutive_indices.append(v)
        else:
            if i % 2 == 0:
                if v > indices[i - 1] + 1:
                    consecutive_indices.append(v)
            else:
                if v + 1 < indices[i + 1]:
                    consecutive_indices.append(v)
    assert len(consecutive_indices) % 2 == 0 and len(consecutive_indices) <= len(indices)
    if len(indices) != len(consecutive_indices):
        indices = " ".join([str(i) for i in indices])
        print("Convert from [%s] to [%s]." % (indices, " ".join([str(i) for i in consecutive_indices])))
    return consecutive_indices


def load_token_boundaries(filepath):
    token_start, token_end = {}, {}
    with open(filepath) as f:
        pre_doc, sent_idx, token_idx = None, 0, 0
        for line in f:
            if len(line.strip()) == 0 and pre_doc is not None:
                sent_idx += 1
                token_idx = 0
                continue
            sp = line.strip().split()
            assert len(sp) == 4
            token, doc, start, end = sp
            if pre_doc is None:
                pre_doc = doc
            if pre_doc != doc:
                sent_idx = 0
                assert token_idx == 0
                pre_doc = doc
            token_start[(doc, int(start))] = (sent_idx, token_idx, token)
            token_end[(doc, int(end))] = (sent_idx, token_idx, token)
            token_idx += 1
    return token_start, token_end


def find_token_starting_at_offset(doc, offset, token_boundaries):
    if (doc, offset) in token_boundaries: return token_boundaries[(doc, offset)]

    adjust = 0
    while (offset - adjust) >= 0 and (doc, offset - adjust) not in token_boundaries:
        adjust += 1

    if (doc, offset - adjust) in token_boundaries:
        print("Cannot find original offset (%d) in document (%s), so use (%d) instead." % (offset, doc, offset - adjust))
        return token_boundaries[(doc, offset - adjust)]
    else:
        print("Cannot find offset (%d) in document (%s)." % (offset, doc))
        return None


def find_token_ending_at_offset(doc, offset, token_boundaries):
    if (doc, offset) in token_boundaries: return token_boundaries[(doc, offset)]

    adjust = 0
    while adjust < 20 and (doc, offset + adjust) not in token_boundaries:
        adjust += 1

    if (doc, offset + adjust) in token_boundaries:
        print("Cannot find original offset (%d) in document (%s), so use (%d) instead." % (offset, doc, offset + adjust))
        return token_boundaries[(doc, offset + adjust)]
    else:
        print("Cannot find offset (%d) in document (%s)." % (offset, doc))
        return None


if __name__ == "__main__":
    args = parse_parameters()
    token_start, token_end = load_token_boundaries(args.input_tokens)

    with open(args.output_ann, "w") as out_f:
        with open(args.input_ann) as in_f:
            for line in in_f:
                sp = line.strip().split("\t")
                assert len(sp) == 4
                doc, label, indices, mention = sp
                indices = [int(i) for i in indices.split(",")]

                token_indices = []
                for i in range(0, len(indices), 2):
                    start_token_idx = find_token_starting_at_offset(doc, indices[i], token_start)
                    end_token_idx = find_token_ending_at_offset(doc, indices[i + 1], token_end)
                    assert start_token_idx is not None and end_token_idx is not None
                    token_indices.append(start_token_idx)
                    token_indices.append(end_token_idx)

                assert len(indices) == len(token_indices)
                assert len(set([i[0] for i in token_indices])) == 1

                sent_idx = token_indices[0][0]
                token_indices = sorted([i[1] for i in token_indices])
                token_indices = merge_consecutive_indices(token_indices)
                token_indices = ",".join([str(i) for i in token_indices])
                out_f.write("%s\t%s\t%s\t%s\t%s\n" % (doc, sent_idx, label, token_indices, mention))