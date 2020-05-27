import itertools, torch
from typing import List


'''Update date: 2019-Nov-5'''
def block_orthogonal(tensor, split_sizes: List[int], gain=1.0):
    # tensor: the tensor to initialize
    # split_sizes: [10, 20] result in the tensor being split into chunks of size 10 along the first dimension
    # 20 along the second
    # Used in the case of recurrent models which use multiple gates applied to linear projections.
    # Separate parameters should be initialized independently
    data = tensor.data
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("block_orthogonal: tensor size and split sizes not compatible!")

    indexes = [list(range(0, max_size, split)) for max_size, split in zip(sizes, split_sizes)]
    for block_state_indices in itertools.product(*indexes):
        index_and_step_tuples = zip(block_state_indices, split_sizes)
        block_slice = tuple([slice(start_index, start_index + step) for start_index, step in index_and_step_tuples])
        data[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#clamp_tensor
Update date: 2019-Nov-6'''
def clamp_tensor(tensor, minimum, maximum):
    if tensor.is_sparse:
        coalesced_tensor = tensor.coalesce()

        coalesced_tensor._values().clamp_(minimum, maximum)
        return coalesced_tensor
    else:
        return tensor.clamp(minimum, maximum)


'''Update date: 2019-Nov-7'''
def enable_gradient_clipping(model, grad_clipping) -> None:
    if grad_clipping is not None:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad: clamp_tensor(grad, minimum=-grad_clipping, maximum=grad_clipping))


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/highway.py
https://github.com/LiyuanLucasLiu/LM-LSTM-CRF/blob/master/model/highway.py
A gated combination of a linear transformation and a non-linear transformation of its input.
Math: y = g * x + (1 - g) * f(A(x)),
g is an element-wise gate, computed as: sigmoid(B(x)).
A is a linear transformation, f is an element-wise non-linearity
Update date: 2019-Nov-5'''
class Highway(torch.nn.Module):
    def __init__(self, input_dim, num_layers=1):
        super(Highway, self).__init__()

        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        for layer in self._layers:
            # Bias the highway layer to just carry its input forward.
            # Set the bias on B(x) to be positive, then g will be biased to be high
            # The bias on B(x) is the second half of the bias vector in each linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_inputs = inputs
        for layer in self._layers:
            linear_part = current_inputs
            projected_inputs = layer(current_inputs)

            nonlinear_part, gate = projected_inputs.chunk(2, dim=-1)
            nonlinear_part = torch.nn.functional.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_inputs = gate * linear_part + (1 - gate) * nonlinear_part
        return current_inputs


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
Update date: 2019-April-26'''
def masked_softmax(vector, mask=None):
    if mask is None:
        return torch.nn.functional.softmax(vector, dim=-1)
    else:
        mask = mask.float()
        assert mask.dim() == vector.dim()
        # use a very large negative number for those masked positions
        # so that the probabilities of those positions would be approximately 0.
        # This is not accurate in math, but works for most cases and consumes less memory.
        masked_vector = vector.masked_fill((1 - mask).byte(), -1e32)
        return torch.nn.functional.softmax(masked_vector, dim=-1)


'''Update date: 2019-Nov-7'''
def rescale_gradients(model, grad_norm):
    if grad_norm:
        parameters = [p for p in model.parameters() if p.grad is not None]
        return sparse_clip_norm(parameters, grad_norm)
    return None


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py
Compute a parameterised scalar mixture of N tensors:
        outs = gamma * sum(s_k * tensor_k)
        s_k = softmax(w)
        gamma and w are parameters
        Imagine tensor_k are outputs of each layer in ELMo, and outs is its final weighted (s_k) representation.
Update date: 2019-Nov-5'''
class ScalarMix(torch.nn.Module):
    def __init__(self, num_tensors, trainable=True):
        super(ScalarMix, self).__init__()
        self.num_tensors = num_tensors
        self.scalar_parameters = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad=trainable) for _ in range(num_tensors)])
        self.gamma = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors, mask=None):
        # tensors must all be the same shape, let's say (batch_size, timesteps, dim)
        assert self.num_tensors == len(tensors)

        normed_weights = torch.nn.functional.softmax(torch.cat([p for p in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


'''Update date: 2019-Nov-7'''
def sparse_clip_norm(parameters, max_norm: float):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        if p.grad.is_sparse:
            grad = p.grad.data.coalesce()
            param_norm = grad._values().norm(2.)
        else:
            param_norm = p.grad.data.norm(2.)

        total_norm += param_norm ** 2.

    total_norm = total_norm ** (1. / 2.)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad.is_sparse:
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/time_distributed.py
Given an input shaped like (batch_size, sequence_length, ...) and a Module that takes input like (batch_size, ...)
TimeDistributed can reshape the input to be (batch_size * sequence_length, ...) applies the Module, then reshape back.
Update date: 2019-Nov-5'''
class TimeDistributed(torch.nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self._module = module

    def forward(self, *inputs):
        reshaped_inputs = []

        for input_tensor in inputs:
            input_size = input_tensor.size()
            assert len(input_size) > 2
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

        reshaped_outputs = self._module(*reshaped_inputs)

        original_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
        outputs = reshaped_outputs.contiguous().view(*original_shape)
        return outputs