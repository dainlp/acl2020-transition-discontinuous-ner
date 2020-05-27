import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
inputs:
    tensor: batch first tensor
outputs:
    sorted_sequence_lengths: sorted by decreasing size
    restoration_indices: sorted_tensor.index_select(0, restoration_indices) == original_tensor
    permutation_index: useful if want to sort other tensors using the same ordering
Update date: 2019-April-26'''
def sort_batch_by_length(tensor, sequence_lengths):
    assert isinstance(tensor, torch.Tensor) and isinstance(sequence_lengths, torch.Tensor)

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)

    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/encoder_base.py
Update date: 2019-03-03'''
class EncoderBase(torch.nn.Module):
    def __init__(self):
        super(EncoderBase, self).__init__()


    def sort_and_run_forward(self, module, inputs, mask):
        sequence_lengths = mask.long().sum(-1)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, sorting_indices = sort_batch_by_length(inputs,
                                                                                                            sequence_lengths)

        packed_sequence_input = pack_padded_sequence(sorted_inputs[:, :, :],
                                                     sorted_sequence_lengths[:].data.tolist(),
                                                     batch_first=True)

        module_output, final_states = module(packed_sequence_input, None)
        return module_output, final_states, restoration_indices


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2seq_encoders/__init__.py
Update date: 2019-03-03'''
class LstmEncoder(EncoderBase):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, bidirectional=True):
        super(LstmEncoder, self).__init__()

        self._module = torch.nn.LSTM(batch_first=True, input_size=input_size, hidden_size=hidden_size,
                                     num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)


    def forward(self, inputs, mask=None):
        packed_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._module, inputs, mask)
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
        return unpacked_sequence_tensor.index_select(0, restoration_indices)