'''Update date: 2020-Jan-13'''
import argparse
from typing import List


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


class Span(object):
    def __init__(self, start, end):
        '''start and end are inclusive'''
        self.start = int(start)
        self.end = int(end)


    @classmethod
    def overlaps(cls, span1, span2):
        '''whether span1 overlaps with span2, including equals'''
        if span1.end < span2.start: return False
        if span1.start > span2.end: return False
        return True


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        return "%d,%d" % (self.start, self.end)


class Mention(object):
    def __init__(self, spans, label: str):
        assert len(spans) >= 1
        self.spans = spans
        self.label = label
        self.discontinuous = (len(spans) > 1)
        self._overlapping_spans = set()


    @property
    def start(self):
        return self.spans[0].start


    @property
    def end(self):
        return self.spans[-1].end


    @property
    def overlapping(self):
        return len(self._overlapping_spans) > 0


    @classmethod
    def overlap_spans(cls, mention1, mention2):
        for span1 in mention1.spans:
            for span2 in mention2.spans:
                if Span.overlaps(span1, span2):
                    return True
        return False


    @classmethod
    def remove_discontinuous_mentions(cls, mentions):
        '''convert discontinuous mentions, such as 17,20,22,22 Disorder, to 17,22 Disorder'''
        continuous_mentions = []
        for mention in mentions:
            if mention.discontinuous:
                continuous_mentions.append(Mention.create_mention([mention.start, mention.end], mention.label))
            else:
                continuous_mentions.append(mention)
        return continuous_mentions


    @classmethod
    def merge_overlapping_mentions(cls, mentions):
        '''
        Given a list of mentions which may overlap with each other, erase these overlapping.
        For example
            1) if an mention starts at 1, ends at 4, the other one starts at 3, ends at 5.
            Then group these together as one mention starting at 1, ending at 5 if they are of the same type,
                otherwise, raise an Error.
        '''
        overlapping_may_exist = True
        while overlapping_may_exist:
            overlapping_may_exist = False
            merged_mentions = {}
            for i in range(len(mentions)):
                for j in range(len(mentions)):
                    if i == j: continue
                    if Mention.overlap_spans(mentions[i], mentions[j]):
                        assert mentions[i].label == mentions[j].label, "TODO: two mentions of different types overlap"
                        overlapping_may_exist = True
                        merged_mention_start = min(mentions[i].start, mentions[j].start)
                        merged_mention_end = max(mentions[i].end, mentions[j].end)
                        merged_mention = Mention.create_mention([merged_mention_start, merged_mention_end],
                                                                mentions[i].label)
                        if (merged_mention_start, merged_mention_end) not in merged_mentions:
                            merged_mentions[(merged_mention_start, merged_mention_end)] = merged_mention
                        mentions[i]._overlapping_spans.add(0)
                        mentions[j]._overlapping_spans.add(0)
            mentions = [mention for mention in mentions if not mention.overlapping] + list(merged_mentions.values())
        return mentions


    @classmethod
    def create_mention(cls, indices, label: str):
        '''
        the original indices can be 136,142,143,147, these two spans are actually consecutive, so convert to 136,147
                similarily, convert 136,142,143,147,148,160 into 136,160 (these three spans are consecutive)
        additionally, sort the indices: 119,125,92,96 to 92,96,119,125
        '''
        assert len(indices) % 2 == 0
        indices = sorted(indices)
        indices = merge_consecutive_indices(indices)
        spans = [Span(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]
        return cls(spans, label)


    @classmethod
    def create_mentions(cls, mentions: str):
        '''Input: 5,6 DATE|6,6 DAY|5,6 EVENT'''
        if len(mentions.strip()) == 0: return []
        results = []
        for mention in mentions.split("|"):
            indices, label = mention.split()
            indices = [int(i) for i in indices.split(",")]
            results.append(Mention.create_mention(indices, label))
        return results


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        spans = [str(s) for s in self.spans]
        return "%s %s" % (",".join(spans), self.label)


def parse_parameters(parser=None):
    if parser is None: parser = argparse.ArgumentParser()

    ## Required
    parser.add_argument("--input_filepath", type=str)
    parser.add_argument("--output_filepath", type=str)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_parameters()

    num_flat_mentions, num_original_mentions = 0, 0

    with open(args.output_filepath, "w") as out_f:
        with open(args.input_filepath) as in_f:
            for text in in_f:
                out_f.write(text)
                mentions = next(in_f).strip()
                assert len(next(in_f).strip()) == 0
                if len(mentions) > 0:
                    mentions = Mention.create_mentions(mentions)
                    num_original_mentions += len(mentions)
                    disc_removed = Mention.remove_discontinuous_mentions(mentions)
                    flat_mentions = Mention.merge_overlapping_mentions(disc_removed)
                    num_flat_mentions += len(flat_mentions)
                    mentions = "|".join([str(m) for m in flat_mentions])
                out_f.write("%s\n\n" % mentions)

    print("After merging overlapping mentions, there are %d out of %d mentions." % (num_flat_mentions, num_original_mentions))