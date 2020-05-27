import torch
import torch.nn.functional as F
from xdai.utils.attention import BilinearAttention
from xdai.utils.instance import TextField
from xdai.utils.seq2seq import LstmEncoder
from xdai.utils.token_embedder import Embedding, TextFieldEmbedder
from xdai.ner.transition_discontinuous.parsing import Parser


class _Buffer:
    def __init__(self, sentence, empty_state):
        sentence_length, _ = sentence.size() # sentence length, contextual word embedding size
        self.sentence_length = sentence_length
        self.sentence = sentence
        self.pointer = 0
        self.empty_state = empty_state


    def top(self):  # return the first word representation
        if self.pointer < self.sentence_length:
            return self.sentence[self.pointer]
        else:
            return self.empty_state


    def pop(self): # move out the first word, and return its word representation
        assert self.pointer < self.sentence_length
        self.pointer += 1
        return self.sentence[self.pointer - 1]


    def __len__(self):
        return self.sentence_length - self.pointer


class _StackLSTM:
    def __init__(self, lstm_cell, initial_state):
        self.lstm_cell = lstm_cell
        self.state = [initial_state]


    def push(self, input):
        self.state.append(self.lstm_cell(input.unsqueeze(0), self.state[-1]))


    def pop(self):
        assert len(self.state) > 1
        return self.state.pop()[0].squeeze(0)


    def top(self):
        assert len(self.state) > 0
        return self.state[-1][0].squeeze(0)


    def top3(self):
        if len(self) >= 3:
            return (self.state[-3][0].squeeze(0), self.state[-2][0].squeeze(0), self.state[-1][0].squeeze(0))
        elif len(self) >= 2:
            return (self.state[-2][0].squeeze(0), self.state[-2][0].squeeze(0), self.state[-1][0].squeeze(0))
        else:
            return (self.state[-1][0].squeeze(0), self.state[-1][0].squeeze(0), self.state[-1][0].squeeze(0))


    def __len__(self):
        return len(self.state) - 1


class _LeafModule(torch.nn.Module):
    def __init__(self, input_linear, output_linear):
        super(_LeafModule, self).__init__()
        self.input_linear = input_linear
        self.output_linear = output_linear


    def forward(self, inputs):
        cell_state = self.input_linear(inputs)
        hidden_state = torch.sigmoid(self.output_linear(inputs)) * torch.tanh(cell_state)
        return hidden_state, cell_state


class _ReduceModule(torch.nn.Module):
    def __init__(self, reduce_linears):
        super(_ReduceModule, self).__init__()
        self.reduce_linears = reduce_linears


    def forward(self, left_cell, left_hidden, right_cell, right_hidden):
        concate_hidden = torch.cat([left_hidden, right_hidden], 0)
        input_gate = torch.sigmoid(self.reduce_linears[0](concate_hidden))
        left_gate = torch.sigmoid(self.reduce_linears[1](concate_hidden))
        right_gate = torch.sigmoid(self.reduce_linears[2](concate_hidden))
        candidate_cell_state = torch.tanh(self.reduce_linears[3](concate_hidden))
        cell_state = input_gate * candidate_cell_state + left_gate * left_cell + right_gate * right_cell
        hidden_state = torch.tanh(cell_state)
        return hidden_state, cell_state


class _Stack(object):
    def __init__(self, lstm_cell, initial_state, input_linear, output_linear, reduce_linears):
        self.stack_lstm = _StackLSTM(lstm_cell, initial_state)
        self.leaf_module = _LeafModule(input_linear, output_linear)
        self.reduce_module = _ReduceModule(reduce_linears)
        self._states = []


    def shift(self, input):
        hidden, cell = self.leaf_module(input)
        self._states.append((hidden, cell))
        self.stack_lstm.push(hidden)


    def reduce(self, keep=None):
        assert len(self._states) > 1
        right_hidden, right_cell = self._states.pop()
        left_hidden, left_cell = self._states.pop()
        hidden, cell = self.reduce_module(left_cell, left_hidden, right_cell, right_hidden)

        if keep == "RIGHT":
            self._states.append((right_hidden, right_cell))
            self.stack_lstm.state.pop(-2)
        elif keep == "LEFT":
            self._states.append((left_hidden, left_cell))
            self.stack_lstm.state.pop(-1)
        else:
            self.stack_lstm.state.pop()
            self.stack_lstm.state.pop()

        self._states.append((hidden, cell))
        self.stack_lstm.push(hidden)


    def top(self):
        return self.stack_lstm.top()


    def top3(self):
        return self.stack_lstm.top3()


    def pop(self):
        self._states.pop()
        self.stack_lstm.pop()


    def __len__(self):
        return len(self._states)


def _xavier_initialization(*size):
    p = torch.nn.init.xavier_normal_(torch.cuda.FloatTensor(*size)).cuda()
    return torch.nn.Parameter(p)


'''Update at 2019-Nov-7'''
class TransitionModel(torch.nn.Module):
    def __init__(self, args, vocab):
        super(TransitionModel, self).__init__()
        self.idx2action = vocab.get_index_to_item_vocabulary("actions")
        self.action2idx = vocab.get_item_to_index_vocabulary("actions")

        self.text_filed_embedder = TextFieldEmbedder.create_embedder(args, vocab)
        self.action_embedding = Embedding(vocab.get_vocab_size("actions"), args.action_embedding_size)
        self.encoder = LstmEncoder(input_size=self.text_filed_embedder.get_output_dim(),
                                   hidden_size=args.lstm_cell_size,
                                   num_layers=args.lstm_layers, dropout=args.dropout, bidirectional=True)
        self.dropout = torch.nn.Dropout(args.dropout)
        self.token_empty = torch.nn.Parameter(torch.randn(args.lstm_cell_size * 2))

        self.stack_lstm = torch.nn.LSTMCell(args.lstm_cell_size * 2, args.lstm_cell_size * 2)
        self.stack_lstm_initial = (
        [_xavier_initialization(1, args.lstm_cell_size * 2), _xavier_initialization(1, args.lstm_cell_size * 2)])

        self.action_lstm = torch.nn.LSTMCell(args.action_embedding_size, args.lstm_cell_size * 2)
        self.action_lstm_initial = (
        [_xavier_initialization(1, args.lstm_cell_size * 2), _xavier_initialization(1, args.lstm_cell_size * 2)])

        self.leaf_input_linear = torch.nn.Linear(args.lstm_cell_size * 2, args.lstm_cell_size * 2)
        self.leaf_output_linear = torch.nn.Linear(args.lstm_cell_size * 2, args.lstm_cell_size * 2)
        self.reduce_module = torch.nn.ModuleList(
            [torch.nn.Linear(2 * args.lstm_cell_size * 2, args.lstm_cell_size * 2) for _ in range(4)])

        self.hidden2feature = torch.nn.Linear(args.lstm_cell_size * 2 * 8, args.lstm_cell_size)
        self.feature2action = torch.nn.Linear(args.lstm_cell_size, vocab.get_vocab_size("actions"))

        self.stack1_attention = BilinearAttention(args.lstm_cell_size * 2, args.lstm_cell_size * 2)
        self.stack2_attention = BilinearAttention(args.lstm_cell_size * 2, args.lstm_cell_size * 2)
        self.stack3_attention = BilinearAttention(args.lstm_cell_size * 2, args.lstm_cell_size * 2)

        self.parser = Parser()

        self._metric = {"correct_actions": 0, "total_actions": 0,
                        "correct_mentions": 0, "total_gold_mentions": 0, "total_pred_mentions": 0,
                        "correct_disc_mentions": 0, "total_gold_disc_mentions": 0, "total_pred_disc_mentions": 0}


    def _get_possible_actions(self, stack, buffer, previous_action_name=""):
        valid_actions = []

        if len(buffer) > 0:
            valid_actions.append(self.action2idx["SHIFT"])
            valid_actions.append(self.action2idx["OUT"])
        if len(stack) >= 1:
            valid_actions += [i for i, a in self.idx2action.items() if a.startswith("COMPLETE")]
            if len(stack) >= 2 and not previous_action_name in ["LEFT-REDUCE", "RIGHT-REDUCE"]:
                valid_actions += [i for i, a in self.idx2action.items() if a.find("REDUCE") >= 0]
        valid_actions = sorted(valid_actions)
        return valid_actions


    def get_metrics(self, reset=False):
        _metrics = {}
        _metrics["accuracy"] = self._metric["correct_actions"] / self._metric["total_actions"] if self._metric[
                                                                                                      "total_actions"] > 0 else 0.0
        _metrics["precision-overall"] = self._metric["correct_mentions"] / self._metric["total_pred_mentions"] if \
            self._metric["total_pred_mentions"] > 0 else 0.0
        _metrics["recall-overall"] = self._metric["correct_mentions"] / self._metric["total_gold_mentions"] if \
            self._metric["total_gold_mentions"] > 0 else 0.0
        _metrics["f1-overall"] = 2 * _metrics["precision-overall"] * _metrics["recall-overall"] / (
                _metrics["precision-overall"] + _metrics["recall-overall"]) if _metrics["precision-overall"] + \
                                                                               _metrics[
                                                                                   "recall-overall"] > 0 else 0.0

        _metrics["discontinuous-precision-overall"] = self._metric["correct_disc_mentions"] / self._metric[
            "total_pred_disc_mentions"] if \
            self._metric["total_pred_disc_mentions"] > 0 else 0.0
        _metrics["discontinuous-recall-overall"] = self._metric["correct_disc_mentions"] / self._metric[
            "total_gold_disc_mentions"] if self._metric[
                                               "total_gold_disc_mentions"] > 0 else 0.0
        _metrics["discontinuous-f1-overall"] = 2 * _metrics["discontinuous-precision-overall"] * _metrics[
            "discontinuous-recall-overall"] / (_metrics["discontinuous-precision-overall"] + _metrics[
            "discontinuous-recall-overall"]) if \
            _metrics["discontinuous-precision-overall"] + _metrics["discontinuous-recall-overall"] > 0 else 0.0

        if reset:
            self._metric = {"correct_actions": 0, "total_actions": 0, "correct_mentions": 0,
                            "total_gold_mentions": 0, "total_pred_mentions": 0, "correct_disc_mentions": 0,
                            "total_gold_disc_mentions": 0, "total_pred_disc_mentions": 0}

        return _metrics


    def _build_state_representation(self, buffer, stack, action_history):
        top3_stack, top2_stack, top1_stack = stack.top3()
        buffer_outputs = torch.unsqueeze(buffer.sentence, 0)
        top1_attn_weights = self.stack1_attention(torch.unsqueeze(top1_stack, 0), buffer_outputs)
        top1_attn_applied = torch.bmm(top1_attn_weights.unsqueeze(0), buffer_outputs).squeeze()
        top2_attn_weights = self.stack2_attention(torch.unsqueeze(top2_stack, 0), buffer_outputs)
        top2_attn_applied = torch.bmm(top2_attn_weights.unsqueeze(0), buffer_outputs).squeeze()
        top3_attn_weights = self.stack3_attention(torch.unsqueeze(top3_stack, 0), buffer_outputs)
        top3_attn_applied = torch.bmm(top3_attn_weights.unsqueeze(0), buffer_outputs).squeeze()
        features = torch.cat(
            [top1_stack, top1_attn_applied, top2_stack, top2_attn_applied, top3_stack, top3_attn_applied,
             buffer.top(), action_history.top()], 0)
        return features


    def _apply_action(self, stack, buffer, action_name):
        if action_name == "SHIFT":
            stack.shift(buffer.pop().cuda())
        elif action_name.startswith("OUT"):
            buffer.pop()
        elif action_name.startswith("COMPLETE"):
            stack.pop()
        elif action_name.find("REDUCE") >= 0:
            if action_name.startswith("LEFT"):
                stack.reduce(keep="LEFT")
            elif action_name.startswith("RIGHT"):
                stack.reduce(keep="RIGHT")
            else:
                stack.reduce()
        else:
            raise ValueError
        return stack, buffer


    def forward(self, tokens, actions, annotations, **kwargs):
        embedded_tokens = self.dropout(self.text_filed_embedder(tokens))
        mask = TextField.get_text_field_mask(tokens)
        sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
        encoded_tokens = self.dropout(self.encoder(embedded_tokens, mask))

        batch_size, _, _ = encoded_tokens.size()

        total_loss = 0
        preds = []
        for i in range(batch_size):
            gold_actions, pred_actions = [], []

            stack = _Stack(self.stack_lstm, self.stack_lstm_initial, self.leaf_input_linear, self.leaf_output_linear,
                           self.reduce_module)
            action_history = _StackLSTM(self.action_lstm, self.action_lstm_initial)
            buffer = _Buffer(encoded_tokens[i][0:sequence_lengths[i]], self.token_empty)
            previous_action_name = ""

            if self.training:
                for j in range(len(actions[i])):
                    gold_action = actions[i][j]
                    if type(gold_action) != int: gold_action = gold_action.cpu().data.numpy().item()
                    if gold_action == 0: break

                    valid_actions = self._get_possible_actions(stack, buffer, previous_action_name)
                    assert gold_action in valid_actions
                    gold_actions.append(gold_action)
                    gold_action_name = self.idx2action[gold_action]
                    previous_action_name = gold_action_name

                    if len(valid_actions) == 1:
                        pred_action = valid_actions[0]
                        assert pred_action == gold_action
                    else:
                        features = self._build_state_representation(buffer, stack, action_history)
                        features = F.relu(self.hidden2feature(self.dropout(features)))
                        logits = self.feature2action(features)[torch.LongTensor(valid_actions).cuda()]
                        log_probs = torch.nn.functional.log_softmax(logits, 0)
                        pred_action = valid_actions[torch.max(logits.cpu(), 0)[1].data.numpy().item()]

                        valid_actions = {a: i for i, a in enumerate(valid_actions)}
                        assert len(log_probs) == len(valid_actions)
                        total_loss += log_probs[valid_actions[gold_action]]

                    pred_actions.append(pred_action)
                    action_history.push(
                        self.dropout(self.action_embedding(torch.LongTensor([gold_action]).cuda())).squeeze(0))

                    stack, buffer = self._apply_action(stack, buffer, gold_action_name)

                    assert len(gold_actions) == len(pred_actions)
                    for g, p in zip(gold_actions, pred_actions):
                        if g == p:
                            self._metric["correct_actions"] += 1
                        self._metric["total_actions"] += 1
            else:
                while True:
                    valid_actions = self._get_possible_actions(stack, buffer, previous_action_name)
                    if len(valid_actions) < 1:
                        break
                    elif len(valid_actions) == 1:
                        pred_action = valid_actions[0]
                    else:
                        features = self._build_state_representation(buffer, stack, action_history)
                        features = F.relu(self.hidden2feature(self.dropout(features)))

                        logits = self.feature2action(features)[torch.LongTensor(valid_actions).cuda()]
                        pred_action = valid_actions[torch.max(logits.cpu(), 0)[1].data.numpy().item()]

                    pred_actions.append(pred_action)
                    pred_action_name = self.idx2action[pred_action]
                    previous_action_name = pred_action_name

                    action_history.push(
                        self.dropout(self.action_embedding(torch.LongTensor([pred_action]).cuda())).squeeze(0))

                    stack, buffer = self._apply_action(stack, buffer, pred_action_name)

                gold_mentions = annotations[i].split("|") if len(annotations[i].strip()) > 0 else []
                discontinuous = [1 if len(m.split(" ")[0].split(",")) > 2 else 0 for m in gold_mentions]
                discontinuous = sum(discontinuous) > 0
                pred_actions = [self.idx2action[p] for p in pred_actions]
                pred_mentions = self.parser.parse(pred_actions)
                pred_mentions = [str(p) for p in pred_mentions]
                for p in pred_mentions:
                    if p in gold_mentions:
                        self._metric["correct_mentions"] += 1
                        if discontinuous:
                            self._metric["correct_disc_mentions"] += 1
                self._metric["total_gold_mentions"] += len(gold_mentions)
                self._metric["total_pred_mentions"] += len(pred_mentions)
                if discontinuous:
                    self._metric["total_gold_disc_mentions"] += len(gold_mentions)
                    self._metric["total_pred_disc_mentions"] += len(pred_mentions)
                preds.append("|".join(pred_mentions))

        return {"loss": -1.0 * total_loss, "preds": preds}