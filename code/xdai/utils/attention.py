import torch
from xdai.utils.nn import masked_softmax


class Attention(torch.nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self._normalize = normalize


    def forward(self, vector, matrix, matrix_mask=None):
        similarities = self._forward_internal(vector, matrix)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities


    def _forward_internal(self, vector, matrix):
        raise NotImplementedError


class BilinearAttention(Attention):
    '''The similarity between the vector x and the matrix y is: x^T W y + b, where W, b are parameters'''
    def __init__(self, vector_dim, matrix_dim):
        super().__init__()
        self._W = torch.nn.parameter.Parameter(torch.Tensor(vector_dim, matrix_dim))
        self._b = torch.nn.parameter.Parameter(torch.Tensor(1))
        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._W)
        self._b.data.fill_(0)


    def _forward_internal(self, vector, matrix):
        intermediate = vector.mm(self._W).unsqueeze(1)
        return intermediate.bmm(matrix.transpose(1, 2)).squeeze(1) + self._b