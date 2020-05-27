import h5py, json, torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from xdai.utils.common import sort_batch_by_length
from xdai.utils.nn import block_orthogonal, Highway, ScalarMix
from xdai.utils.token_indexer import ELMoCharacterMapper
from xdai.elmo.utils import add_sentence_boundary_token_ids, remove_sentence_boundaries


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py
Update date: 2019-Nov-5'''
class _ElmoCharacterEncoder(torch.nn.Module):
    '''{"char_cnn":
            {"activation": "relu", "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
            "n_highway": 2, "embedding": {"dim": 16},
            "n_characters": 262, "max_characters_per_token": 50}}'''
    def __init__(self, options_file, weight_file, requires_grad=False):
        super(_ElmoCharacterEncoder, self).__init__()

        with open(options_file, "r") as f:
            self._options = json.load(f)

        self._weight_file = weight_file
        self.output_dim = self._options["lstm"]["projection_dim"]
        self._requires_grad = requires_grad
        self._load_weights()

        self._beginning_of_sentence = torch.from_numpy(
            np.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1)
        self._end_of_sentence = torch.from_numpy(
            np.array(ELMoCharacterMapper.end_of_sentence_characters) + 1)


    def get_output_dim(self):
        return self.output_dim


    def forward(self, inputs):
        '''inputs: batch_size, num_tokens, 50
        return: batch_size, num_tokens + 2, output_dim'''
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        inputs_with_boundary, mask_with_boundary = add_sentence_boundary_token_ids(inputs, mask,
                                                                                    self._beginning_of_sentence,
                                                                                    self._end_of_sentence)
        max_chars_per_token = self._options["char_cnn"]["max_characters_per_token"]
        character_embedding = torch.nn.functional.embedding(inputs_with_boundary.view(-1, max_chars_per_token),
                                                            self._char_embedding_weights)

        assert self._options["char_cnn"]["activation"] == "relu"

        # shape after transpose: (batch_size * (num_tokens + 2), output_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, "char_conv_%d" % i)
            convolved = conv(character_embedding)
            # for each width, (batch_size * (num_tokens + 2), n_filters)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = torch.nn.functional.relu(convolved)
            convs.append(convolved)

        token_embedding = torch.cat(convs, dim=-1)
        token_embedding = self._highways(token_embedding)
        token_embedding = self._projection(token_embedding)
        batch_size, sequence_length_with_boundary, _ = inputs_with_boundary.size()
        return {"mask": mask_with_boundary,
                "token_embedding": token_embedding.view(batch_size, sequence_length_with_boundary, -1)}


    def _load_weights(self):
        with h5py.File(self._weight_file, "r") as f:
            self._load_char_embedding(f)
            self._load_cnn_weights(f)
            self._load_highway(f)
            self._load_projection(f)


    def _load_char_embedding(self, f):
        char_embedding_weights = f["char_embed"][...]
        weights = np.zeros((char_embedding_weights.shape[0] + 1, char_embedding_weights.shape[1]), dtype="float32")
        weights[1:, :] = char_embedding_weights
        self._char_embedding_weights = torch.nn.Parameter(torch.FloatTensor(weights), requires_grad=self._requires_grad)


    def _load_cnn_weights(self, f):
        filters = self._options["char_cnn"]["filters"]

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(in_channels=self._options["char_cnn"]["embedding"]["dim"], out_channels=num,
                                   kernel_size=width, bias=True)
            weight = f["CNN"]["W_cnn_%d" % i][...]
            bias = f["CNN"]["b_cnn_%d" % i][...]

            weight_reshaped = np.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            assert weight_reshaped.shape == tuple(conv.weight.data.shape)
            conv.weight.data.copy_(torch.FloatTensor(weight_reshaped))
            conv.weight.requires_grad = self._requires_grad
            conv.bias.data.copy_(torch.FloatTensor(bias))
            conv.bias.requires_grad = self._requires_grad

            convolutions.append(conv)
            self.add_module("char_conv_%d" % i, conv)

        self._convolutions = convolutions


    def _load_highway(self, f):
        n_filters = sum(n[1] for n in self._options["char_cnn"]["filters"])
        n_highway = self._options["char_cnn"]["n_highway"]

        self._highways = Highway(n_filters, n_highway)
        for i in range(n_highway):
            # transpose and -1.0 are due to the difference between tensorflow and pytorch
            # tf.matmul(X, W) vs. torch.matmul(W, X)
            # g * x + (1 - g) * f(x) in AllenNLP vs (1 - g) * x + g * f(x) in tf
            w_transform = np.transpose(f["CNN_high_%d" % i]["W_transform"][...])
            w_carry = -1.0 * np.transpose(f["CNN_high_%d" % i]["W_carry"][...])
            weight = np.concatenate([w_transform, w_carry], axis=0)
            self._highways._layers[i].weight.data.copy_(torch.FloatTensor(weight))
            self._highways._layers[i].weight.requires_grad = self._requires_grad

            b_transform = f["CNN_high_%d" % i]["b_transform"][...]
            b_carry = -1.0 * f["CNN_high_%d" % i]["b_carry"][...]
            bias = np.concatenate([b_transform, b_carry], axis=0)
            self._highways._layers[i].bias.data.copy_(torch.FloatTensor(bias))
            self._highways._layers[i].bias.requires_grad = self._requires_grad


    def _load_projection(self, f):
        n_filters = sum(n[1] for n in self._options["char_cnn"]["filters"])
        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        weight = f["CNN_proj"]["W_proj"][...]
        bias = f["CNN_proj"]["b_proj"][...]
        self._projection.weight.data.copy_(torch.FloatTensor(np.transpose(weight)))
        self._projection.weight.requires_grad = self._requires_grad
        self._projection.bias.data.copy_(torch.FloatTensor(bias))
        self._projection.bias.requires_grad = self._requires_grad


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/lstm_cell_with_projection.py
Update date: 2019-Nov-5'''
class _LstmCellWithProjection(torch.nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, go_forward=True,
                 memory_cell_clip=None, hidden_state_clip=None):
        super(_LstmCellWithProjection, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.go_forward = go_forward
        self.memory_cell_clip = memory_cell_clip
        self.hidden_state_clip = hidden_state_clip

        # input gate, forget gate, *, output gate
        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size, bias=True)

        self.state_projection = torch.nn.Linear(cell_size, hidden_size, bias=False)
        self.reset_parameters()


    def reset_parameters(self):
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size].fill_(1.0)


    def forward(self, inputs, batch_lengths, initial_state=None):
        '''inputs: batch_size, num_timesteps, input_size,
        initial_state: a tuple representing the initial hidden state and memory cell of the LSTM
                        (1, batch_size, hidden_size) and (1, batch_size, cell_size)'''
        batch_size, total_timesteps, _ = inputs.size()

        output_accumulator = inputs.new_zeros(batch_size, total_timesteps, self.hidden_size)

        if initial_state is None:
            full_batch_previous_memory = inputs.new_zeros(batch_size, self.cell_size)
            full_batch_previous_state = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)

        current_length_index = batch_size - 1 if self.go_forward else 0

        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            # the batch inputs must be _ordered_ by length from longest (first in batch) to shortest (last)
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                while current_length_index < (len(batch_lengths) - 1) and \
                                batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            timestep_input = inputs[0: current_length_index + 1, index]

            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)


            input_gate = torch.sigmoid(projected_input[:, (0 * self.cell_size):(1 * self.cell_size)] +
                                       projected_state[:, (0 * self.cell_size):(1 * self.cell_size)])
            forget_gate = torch.sigmoid(projected_input[:, (1 * self.cell_size):(2 * self.cell_size)] +
                                        projected_state[:, (1 * self.cell_size):(2 * self.cell_size)])
            memory_init = torch.tanh(projected_input[:, (2 * self.cell_size):(3 * self.cell_size)] +
                                     projected_state[:, (2 * self.cell_size):(3 * self.cell_size)])
            output_gate = torch.sigmoid(projected_input[:, (3 * self.cell_size):(4 * self.cell_size)] +
                                        projected_state[:, (3 * self.cell_size):(4 * self.cell_size)])
            memory = input_gate * memory_init + forget_gate * previous_memory

            if self.memory_cell_clip:
                memory = torch.clamp(memory, -self.memory_cell_clip, self.memory_cell_clip)

            timestep_output = self.state_projection(output_gate * torch.tanh(memory))
            if self.hidden_state_clip:
                timestep_output = torch.clamp(timestep_output, -self.hidden_state_clip, self.hidden_state_clip)

            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

        final_state = (full_batch_previous_state.unsqueeze(0),
                       full_batch_previous_memory.unsqueeze(0))

        return output_accumulator, final_state


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/encoder_base.py
https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo_lstm.py
Update date: 2019-Nov-5'''
class _ElmoLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, num_layers, requires_grad=False,
                 memory_cell_clip=None, hidden_state_clip=None):
        super(_ElmoLSTM, self).__init__()

        self._states = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.requires_grad = requires_grad

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        for i in range(num_layers):
            forward_layer = _LstmCellWithProjection(lstm_input_size, hidden_size, cell_size, go_forward=True,
                                                    memory_cell_clip=memory_cell_clip,
                                                    hidden_state_clip=hidden_state_clip)
            backward_layer = _LstmCellWithProjection(lstm_input_size, hidden_size, cell_size, go_forward=False,
                                                     memory_cell_clip=memory_cell_clip,
                                                     hidden_state_clip=hidden_state_clip)
            lstm_input_size = hidden_size

            self.add_module("forward_layer_%d" % i, forward_layer)
            self.add_module("backward_layer_%d" % i, backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)

        self.forward_layers = forward_layers
        self.backward_layers = backward_layers


    def _sort_and_run_forward(self, module, inputs, mask):
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, 0]).int().item() # just in case some instances may be of zero length.

        sequence_lengths = mask.long().sum(-1)
        sorted_inputs, sorted_sequence_lengths, restoration, sorting = sort_batch_by_length(inputs, sequence_lengths)

        packed_sequence_input = pack_padded_sequence(sorted_inputs[:num_valid, :, :],
                                                     sorted_sequence_lengths[:num_valid].data.tolist(),
                                                     batch_first=True)
        initial_states = self._get_initial_states(batch_size, num_valid, sorting)
        module_output, final_states = module(packed_sequence_input, initial_states)

        return module_output, final_states, restoration


    def _get_initial_states(self, batch_size, num_valid, sorting_indices):
        if self._states is None:
            return None

        if batch_size > self._states[0].size(1):
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            # state has shape (num_layers, batch_size, hidden_size)
            for state in self._states:
                zeros = state.new_zeros(state.size(0), num_states_to_concat, state.size(2))
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            correctly_shaped_states = tuple(state[:, :batch_size, :] for state in self._states)
        else:
            correctly_shaped_states = self._states

        sorted_states = [state.index_select(1, sorting_indices) for state in correctly_shaped_states]
        return tuple(state[:, :num_valid, :].contiguous() for state in sorted_states)


    def _update_states(self, final_states, restoration_indices):
        unsorted_states = [state.index_select(1, restoration_indices) for state in final_states]

        if self._states is None:
            self._states = tuple(state.data for state in unsorted_states)
        else:
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            # Masks for the unused states of shape (1, new_batch_size, 1)
            used_new_rows_mask = [(state[0, :, :].sum(-1) != 0.0).float().view(1, new_state_batch_size, 1) for state in
                                  unsorted_states]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                for old_state, new_state, used_mask in zip(self._states, unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state[:, :new_state_batch_size, :] * (1 - used_mask)
                    old_state[:, :new_state_batch_size, :] = new_state + masked_old_state
                    new_states.append(old_state.detach())
            else:
                new_states = []
                for old_state, new_state, used_mask in zip(self._states, unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state * (1 - used_mask)
                    new_state += masked_old_state
                    new_states.append(new_state.detach())

            self._states = tuple(new_states)


    def reset_states(self):
        self._states = None


    def forward(self, inputs, mask):
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = self._sort_and_run_forward(self._lstm_forward,
                                                                                                inputs, mask)

        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        if num_valid < batch_size:
            zeros = stacked_sequence_output.new_zeros(num_layers, batch_size - num_valid, returned_timesteps,
                                                      encoder_dim)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.new_zeros(num_layers, batch_size, sequence_length_difference,
                                                      stacked_sequence_output[0].size(-1))
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

        self._update_states(final_states, restoration_indices)
        return stacked_sequence_output.index_select(1, restoration_indices)

    def _lstm_forward(self, inputs, initial_state=None):
        if initial_state is None:
            hidden_states = [None] * len(self.forward_layers)
        else:
            assert initial_state[0].size()[0] == len(self.forward_layers)
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, "forward_layer_%d" % i)
            backward_layer = getattr(self, "backward_layer_%d" % i)

            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence, batch_lengths,
                                                                   forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence, batch_lengths,
                                                                      backward_state)

            if i != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence,
                                               backward_output_sequence], -1))

            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1)))

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)

        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple = (torch.cat(final_hidden_states, 0), torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple


    def load_weights(self, weight_file):
        requires_grad = self.requires_grad

        with h5py.File(weight_file, "r") as f:
            for i_layer, lstms in enumerate(zip(self.forward_layers, self.backward_layers)):
                for j_direction, lstm in enumerate(lstms):
                    cell_size = lstm.cell_size

                    dataset = f["RNN_%s" % j_direction]["RNN"]["MultiRNNCell"]["Cell%s" % i_layer]["LSTMCell"]

                    tf_weights = np.transpose(dataset["W_0"][...])
                    torch_weights = tf_weights.copy()

                    input_size = lstm.input_size
                    input_weights = torch_weights[:, :input_size]
                    recurrent_weights = torch_weights[:, input_size:]
                    tf_input_weights = tf_weights[:, :input_size]
                    tf_recurrent_weights = tf_weights[:, input_size:]

                    for torch_w, tf_w in [[input_weights, tf_input_weights], [recurrent_weights, tf_recurrent_weights]]:
                        torch_w[(1 * cell_size):(2 * cell_size), :] = tf_w[(2 * cell_size):(3 * cell_size), :]
                        torch_w[(2 * cell_size):(3 * cell_size), :] = tf_w[(1 * cell_size):(2 * cell_size), :]

                    lstm.input_linearity.weight.data.copy_(torch.FloatTensor(input_weights))
                    lstm.state_linearity.weight.data.copy_(torch.FloatTensor(recurrent_weights))
                    lstm.input_linearity.weight.requires_grad = requires_grad
                    lstm.state_linearity.weight.requires_grad = requires_grad

                    tf_bias = dataset["B"][...]
                    tf_bias[(2 * cell_size):(3 * cell_size)] += 1
                    torch_bias = tf_bias.copy()
                    torch_bias[(1 * cell_size):(2 * cell_size)
                    ] = tf_bias[(2 * cell_size):(3 * cell_size)]
                    torch_bias[(2 * cell_size):(3 * cell_size)
                    ] = tf_bias[(1 * cell_size):(2 * cell_size)]
                    lstm.state_linearity.bias.data.copy_(torch.FloatTensor(torch_bias))
                    lstm.state_linearity.bias.requires_grad = requires_grad

                    proj_weights = np.transpose(dataset["W_P_0"][...])
                    lstm.state_projection.weight.data.copy_(torch.FloatTensor(proj_weights))
                    lstm.state_projection.weight.requires_grad = requires_grad


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py
Update date: 2019-Nov-5'''
class _ElmoBiLSTM(torch.nn.Module):
    def __init__(self, options_file, weight_file, requires_grad=False):
        super(_ElmoBiLSTM, self).__init__()

        self._token_embedder = _ElmoCharacterEncoder(options_file, weight_file, requires_grad=requires_grad)
        with open(options_file, "r") as f:
            options = json.load(f)

        self._elmo_lstm = _ElmoLSTM(input_size=options["lstm"]["projection_dim"],
                                    hidden_size=options["lstm"]["projection_dim"],
                                    cell_size=options["lstm"]["dim"],
                                    num_layers=options["lstm"]["n_layers"],
                                    memory_cell_clip=options["lstm"]["cell_clip"],
                                    hidden_state_clip=options["lstm"]["proj_clip"],
                                    requires_grad=requires_grad)
        self._elmo_lstm.load_weights(weight_file)
        self.num_layers = options["lstm"]["n_layers"] + 1


    def get_output_dim(self):
        return 2 * self._token_embedder.get_output_dim()


    def forward(self, inputs):
        '''inputs: batch_size, num_tokens, 50
        return-activations: batch_size, num_tokens + 2, output_dim
        return-mask: batch_size, num_tokens + 2'''
        token_embedding = self._token_embedder(inputs)
        mask = token_embedding["mask"]
        type_representation = token_embedding["token_embedding"]
        lstm_outputs = self._elmo_lstm(type_representation, mask)

        output_tensors = [
            torch.cat([type_representation, type_representation], dim=-1) * mask.float().unsqueeze(-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))
        return {"activations": output_tensors, "mask": mask}


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py
Update date: 2019-Nov-5'''
class Elmo(torch.nn.Module):
    def __init__(self, options_file, weight_file, num_output_representations=1, requires_grad=False, dropout=0.5):
        super(Elmo, self).__init__()

        self._elmo_lstm = _ElmoBiLSTM(options_file, weight_file, requires_grad=requires_grad)
        self._dropout = torch.nn.Dropout(p=dropout)
        self._scalar_mixes = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(self._elmo_lstm.num_layers)
            self.add_module("scalar_mix_%d" % k, scalar_mix)
            self._scalar_mixes.append(scalar_mix)


    def get_output_dim(self):
        return self._elmo_lstm.get_output_dim()


    def forward(self, inputs):
        assert len(inputs.size()) == 3 # batch_size, num_tokens, 50

        bilstm_output = self._elmo_lstm(inputs)
        layer_activations = bilstm_output["activations"]
        mask_with_sentence_boundary = bilstm_output["mask"]

        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, "scalar_mix_%d" % i)
            representation_with_sentence_bounary = scalar_mix(layer_activations)
            representation_without_sentence_boudary, mask_without_sentence_boundary = remove_sentence_boundaries(
                representation_with_sentence_bounary, mask_with_sentence_boundary)
            representations.append(self._dropout(representation_without_sentence_boudary))

        return {"mask": mask_without_sentence_boundary, "elmo_representations": representations}


    @classmethod
    def load_pretrained_elmo(cls, options_file, weight_file):
        return cls(options_file=options_file, weight_file=weight_file)
