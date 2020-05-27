import torch


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-Nov-5'''
def add_sentence_boundary_token_ids(tensor, mask, sentence_begin_token, sentence_end_token):
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    input_shape = list(tensor.data.shape)
    output_shape = list(input_shape)
    output_shape[1] = input_shape[1] + 2
    tensor_with_boundary_tokens = tensor.new_zeros(*output_shape)
    assert len(input_shape) == 3
    tensor_with_boundary_tokens[:, 1:-1, :] = tensor
    for i, j in enumerate(sequence_lengths):
        tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
        tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
    mask_with_boundary_tokens = ((tensor_with_boundary_tokens > 0).long().sum(dim=-1) > 0).long()

    return tensor_with_boundary_tokens, mask_with_boundary_tokens


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-Nov-5'''
def remove_sentence_boundaries(tensor, mask):
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    input_shape = list(tensor.data.shape)
    output_shape = list(input_shape)
    output_shape[1] = input_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*output_shape)
    output_mask = tensor.new_zeros((output_shape[0], output_shape[1]), dtype=torch.long)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, :(j-2), :] = tensor[i, 1:(j-1), :]
            output_mask[i, :(j-2)] = 1
    return tensor_without_boundary_tokens, output_mask