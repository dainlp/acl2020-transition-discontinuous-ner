import logging
from typing import List
from xdai.ner.mention import merge_consecutive_indices, Mention


logger = logging.getLogger(__name__)


'''Update date: 2019-Nov-5'''
class _NodeInStack(Mention):
    @classmethod
    def single_token_node(cls, idx):
        return Mention.create_mention([idx, idx], "")


    @classmethod
    def merge_nodes(cls, m1, m2):
        # TODO: m1 and m2 cannot completely contain each other
        indices = sorted(m1.indices + m2.indices)
        indices = merge_consecutive_indices(indices)
        return Mention.create_mention(indices, "")



'''Update date: 2019-Nov-5'''
class Parser(object):
    # actions include SHIFT, OUT, COMPLETE-Y, REDUCE, LEFT-REDUCE, RIGHT-REDUCE
    def parse(self, actions, seq_length=None) -> List[Mention]:
        mentions, stack = [], []
        if seq_length is None: seq_length = len(actions) * 2
        buffer = [i for i in range(seq_length)]
        for action in actions:
            if action == "SHIFT":
                if len(buffer) < 1:
                    logger.info("Invalid SHIFT action: the buffer is empty.")
                else:
                    stack.append(_NodeInStack.single_token_node(buffer[0]))
                    buffer.pop(0)
            elif action == "OUT":
                if len(buffer) < 1:
                    logger.info("Invalid OUT action: the buffer is empty.")
                else:
                    buffer.pop(0)
            elif action.startswith("COMPLETE"):
                if len(stack) < 1:
                    logger.info("Invalid COMPLETE action: the stack is empty.")
                else:
                    mention = stack.pop(-1)
                    mention.label = action.split("-")[-1].strip()
                    mentions.append(mention)
            else:
                if action.find("REDUCE") >= 0 and len(stack) >= 2:
                    right_node = stack.pop(-1)
                    left_node = stack.pop(-1)
                    if Mention.contains(left_node, right_node) or Mention.contains(right_node, left_node):
                        logger.info("Invalid REDUCE action: the last two elements in the stack contain each other")
                    else:
                        merged = _NodeInStack.merge_nodes(left_node, right_node)
                        if action.startswith("LEFT"): stack.append(left_node)
                        if action.startswith("RIGHT"): stack.append(right_node)
                        stack.append(merged)
                else:
                    logger.info(
                        "Invalid REDUCE action: %s, the number of elements in the stack is %d." % (action, len(stack)))
        return mentions


    def mention2actions(self, mentions: str, sentence_length: int):
        def _detect_overlapping_mentions(mentions):
            mentions = Mention.create_mentions(mentions)

            for i in range(len(mentions)):
                if mentions[i]._overlapping: continue
                for j in range(len(mentions)):
                    if i == j: continue
                    if Mention.overlap_spans(mentions[i], mentions[j]):
                        assert mentions[i].label == mentions[j].label
                        mentions[i]._overlapping = True
                        mentions[j]._overlapping = True
            return mentions


        def _involve_mention(mentions, token_id):
            for i, mention in enumerate(mentions):
                for span in mention.spans:
                    if span.start <= token_id and token_id <= span.end:
                        return True
            return False


        def _find_relevant_mentions(mentions, node):
            parents, equals = [], []
            for i in range(len(mentions)):
                if Mention.contains(mentions[i], node):
                    parents.append(i)
                    if Mention.equal_spans(mentions[i], node):
                        equals.append(i)
            return parents, equals


        mentions = _detect_overlapping_mentions(mentions)
        actions, stack = [], []
        buffer = [i for i in range(sentence_length)]

        while len(buffer) > 0:
            if not _involve_mention(mentions, buffer[0]):
                actions.append("OUT")
                buffer.pop(0)
            else:
                actions.append("SHIFT")
                stack.append(_NodeInStack.single_token_node(buffer[0]))
                buffer.pop(0)

                stack_changed = True

                # COMPLETE, REDUCE, LEFT-REDUCE, RIGHT-REDUCE
                # if the last item of the stack is a mention, and does not involve with other mentions, then COMPLETE
                while stack_changed:
                    stack_changed = False

                    if len(stack) >= 1:
                        parents, equals = _find_relevant_mentions(mentions, stack[-1])
                        if len(equals) == 1 and len(parents) == 1:
                            actions.append("COMPLETE-%s" % mentions[equals[0]].label)
                            stack.pop(-1)
                            mentions.pop(equals[0])
                            stack_changed = True

                        # three REDUCE actions
                        if len(stack) >= 2:
                            if not Mention.overlap_spans(stack[-2], stack[-1]):
                                last_two_ndoes = _NodeInStack.merge_nodes(stack[-2], stack[-1])
                                parents_of_two, _ = _find_relevant_mentions(mentions, last_two_ndoes)
                                if len(parents_of_two) > 0:
                                    parent_of_left, _ = _find_relevant_mentions(mentions, stack[-2])
                                    parent_of_right, _ = _find_relevant_mentions(mentions, stack[-1])
                                    if len(parents_of_two) != len(parent_of_left):
                                        actions.append("LEFT-REDUCE")
                                        stack.pop(-1)
                                        stack.append(last_two_ndoes)
                                        stack_changed = True
                                    elif len(parents_of_two) != len(parent_of_right):
                                        actions.append("RIGHT-REDUCE")
                                        stack.pop(-2)
                                        stack.append(last_two_ndoes)
                                        stack_changed = True
                                    else:
                                        actions.append("REDUCE")
                                        stack.pop(-1)
                                        stack.pop(-1)
                                        stack.append(last_two_ndoes)
                                        stack_changed = True
        return actions