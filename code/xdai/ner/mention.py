import logging
from typing import List


logger = logging.getLogger(__name__)


'''Update date: 2019-Nov-5'''
class Span(object):
    def __init__(self, start, end):
        '''start and end are inclusive'''
        self.start = int(start)
        self.end = int(end)


    @classmethod
    def contains(cls, span1, span2):
        '''whether span1 contains span2, including equals'''
        return span1.start <= span2.start and span1.end >= span2.end


    @classmethod
    def equals(cls, span1, span2):
        '''whether span1 equals span2'''
        return span1.start == span2.start and span1.end == span2.end


    @classmethod
    def overlaps(cls, span1, span2):
        '''whether span1 overlaps with span2, including equals'''
        if span1.end < span2.start: return False
        if span1.start > span2.end: return False
        return True


    @property
    def length(self):
        return self.end + 1 - self.start


    def __str__(self):
        return "%d,%d" % (self.start, self.end)


'''Update date: 2019-Nov-5'''
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
        logger.debug("Convert from [%s] to [%s]." % (
        " ".join([str(i) for i in indices]), " ".join([str(i) for i in consecutive_indices])))
    return consecutive_indices


'''Update date: 2019-Nov-15'''
class Mention(object):
    def __init__(self, spans: List[Span], label: str):
        assert len(spans) >= 1
        self.spans = spans
        self.label = label

        # assume these spans are not consecutive and sorted by indices, needs to be done before creating the mention
        self.discontinuous = (len(spans) > 1)
        self._overlapping = False
        self._overlapping_spans = set()


    @property
    def start(self):
        return self.spans[0].start


    @property
    def end(self):
        return self.spans[-1].end


    @property
    def indices(self):
        return sorted([span.start for span in self.spans] + [span.end for span in self.spans])


    @property
    def length(self):
        return sum([span.length for span in self.spans])


    @property
    def interval_length(self):
        if self.discontinuous:
            return self.end + 1 - self.start - self.length
        return 0


    @property
    def overlapping(self):
        return len(self._overlapping_spans) > 0


    @property
    def overlap_at_left(self):
        return len(self._overlapping_spans) == 1 and list(self._overlapping_spans)[0] == 0


    @property
    def overlap_at_right(self):
        return len(self._overlapping_spans) == 1 and list(self._overlapping_spans)[0] == len(self.spans) - 1


    @classmethod
    def contains(cls, mention1, mention2):
        span2contained = 0
        for span2 in mention2.spans:
            for span1 in mention1.spans:
                if Span.contains(span1, span2):
                    span2contained += 1
                    break
        return span2contained == len(mention2.spans)


    @classmethod
    def equal_spans(cls, mention1, mention2):
        if len(mention1.spans) != len(mention2.spans): return False
        for span1, span2 in zip(mention1.spans, mention2.spans):
            if not Span.equals(span1, span2):
                return False
        return True


    @classmethod
    def equals(cls, mention1, mention2):
        return Mention.equal_spans(mention1, mention2) and mention1.label == mention2.label


    @classmethod
    def overlap_spans(cls, mention1, mention2):
        overlap_span = False
        for span1 in mention1.spans:
            for span2 in mention2.spans:
                if Span.overlaps(span1, span2):
                    overlap_span = True
                    break
            if overlap_span: break
        return overlap_span


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
    def remove_nested_mentions(cls, mentions):
        '''if an mention is contained completely by one other mention, get rid of the inner one.'''
        outer_mentions = []
        for i in range(len(mentions)):
            nested = False
            for j in range(len(mentions)):
                if i == j: continue
                if Mention.contains(mentions[j], mentions[i]):
                    assert not Mention.contains(mentions[j], mentions[i]), "TODO: multi-type mentions"
                    nested = True
                    break
            if not nested:
                outer_mentions.append(mentions[i])
        return outer_mentions


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
    def create_mention(cls, indices: List[int], label: str):
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
    def create_mentions(cls, mentions: str) -> List[object]:
        '''Input: 5,6 DATE|6,6 DAY|5,6 EVENT'''
        if len(mentions.strip()) == 0: return []
        results = []
        for mention in mentions.split("|"):
            indices, label = mention.split()
            indices = [int(i) for i in indices.split(",")]
            results.append(Mention.create_mention(indices, label))
        return results


    @classmethod
    def check_overlap_spans(cls, mention1, mention2):
        overlap_span = False
        for i, span1 in enumerate(mention1.spans):
            for j, span2 in enumerate(mention2.spans):
                if Span.overlaps(span1, span2):
                    overlap_span = True
                    mention1._overlapping_spans.add(i)
                    mention2._overlapping_spans.add(j)
        return overlap_span


    @classmethod
    def check_overlap_mentions(cls, mentions):
        for i in range(len(mentions)):
            for j in range(len(mentions)):
                if i == j: continue
                Mention.check_overlap_spans(mentions[i], mentions[j])
        return mentions


    def print_text(self, tokens):
        print_tokens = []
        indices = self.indices
        for i, token in enumerate(tokens):
            if i in indices:
                print_tokens.append("\x1b[6;30;42m%s\x1b[0m" % token)
            else:
                print_tokens.append(token)
        print("%s" % " ".join(print_tokens))


    def __str__(self):
        spans = [str(s) for s in self.spans]
        return "%s %s" % (",".join(spans), self.label)


'''Convert a list of BIO tags to a list of mentions
        this list of BIO tags should be in perfect format, for example, I- tag cannot follow a O tag.
        Update: 2019-Nov-1'''
def bio_tags_to_mentions(bio_tags: List[str]) -> List[Mention]:
    mentions = []
    i = 0
    while i < len(bio_tags):
        if bio_tags[i][0] == "B":
            start = i
            end = i
            label = bio_tags[i][2:]
            while end + 1 < len(bio_tags) and bio_tags[end + 1][0] == "I":
                assert bio_tags[end + 1][2:] == label
                end += 1
            mentions.append(Mention.create_mention([start, end], label))
            i = end + 1
        else:
            i += 1
    return mentions


'''Convert a list of BIOES tags to BIO tags
Update: 2019-Oct-13'''
def bioes_to_bio(bioes_tags):
    bio_tags = []
    for tag in bioes_tags:
        if tag[0] == "O":
            bio_tags.append(tag)
        else:
            if tag[0] in ["B", "S"]:
                bio_tags.append("B-%s" % tag[2:])
            else:
                if len(bio_tags) == 0 or bio_tags[-1] == "O":
                    bio_tags.append("B-%s" % tag[2:])
                else:
                    if bio_tags[-1][1:] == tag[1:]:
                        bio_tags.append("I-%s" % tag[2:])
                    else:
                        bio_tags.append("B-%s" % tag[2:])
    assert len(bio_tags) == len(bioes_tags)
    return bio_tags


'''Convert a list of BIO tags to BIOES tags
Update: 2019-Oct-13'''
def bio_to_bioes(original_tags: List[str]) -> List[str]:
    def _change_prefix(original_tag, new_prefix):
        assert original_tag.find("-") > 0 and len(new_prefix) == 1
        chars = list(original_tag)
        chars[0] = new_prefix
        return "".join(chars)

    def _pop_replace_append(stack, bioes_sequence, new_prefix):
        tag = stack.pop()
        new_tag = _change_prefix(tag, new_prefix)
        bioes_sequence.append(new_tag)

    def _process_stack(stack, bioes_sequence):
        if len(stack) == 1:
            _pop_replace_append(stack, bioes_sequence, "S")
        else:
            recoded_stack = []
            _pop_replace_append(stack, recoded_stack, "E")
            while len(stack) >= 2:
                _pop_replace_append(stack, recoded_stack, "I")
            _pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            bioes_sequence.extend(recoded_stack)

    bioes_sequence = []
    stack = []

    for tag in original_tags:
        if tag == "O":
            if len(stack) == 0:
                bioes_sequence.append(tag)
            else:
                _process_stack(stack, bioes_sequence)
                bioes_sequence.append(tag)
        elif tag[0] == "I":
            if len(stack) == 0:
                stack.append(tag)
            else:
                this_type = tag[2:]
                prev_type = stack[-1][2:]
                if this_type == prev_type:
                    stack.append(tag)
                else:
                    _process_stack(stack, bioes_sequence)
                    stack.append(tag)
        elif tag[0] == "B":
            if len(stack) > 0:
                _process_stack(stack, bioes_sequence)
            stack.append(tag)
        else:
            raise ValueError("Invalid tag:", tag)

    if len(stack) > 0:
        _process_stack(stack, bioes_sequence)

    return bioes_sequence


'''Convert a list of mentions into BIO tags: 9,9 Drug|21,21 Drug|11,12 ADR
Update: 2019-Oct-28'''
def mentions_to_bio_tags(mentions: str, num_of_tokens: int):
    tags = ["O"] * num_of_tokens
    if len(mentions.strip()) == 0: return tags
    for mention in mentions.split("|"):
        indices, label = mention.split()
        sp = indices.split(",")
        assert len(sp) == 2, sp
        start, end = int(sp[0]), int(sp[1])
        for i in range(start, end + 1):
            assert tags[i] == "O", mentions
            if i == start:
                tags[i] = "B-%s" % label
            else:
                tags[i] = "I-%s" % label
    return tags