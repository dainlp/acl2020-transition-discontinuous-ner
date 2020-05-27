import torch
import torch.nn.functional as F


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2vec_encoders/cnn_encoder.py
Update date: 2019-Nov-5'''
class CnnEncoder(torch.nn.Module):
    def __init__(self, input_dim=16, num_filters=128, ngram_filter_sizes=[3]):
        super(CnnEncoder, self).__init__()
        self._input_dim = input_dim
        self._convolution_layers = [torch.nn.Conv1d(in_channels=input_dim,
                                                    out_channels=num_filters,
                                                    kernel_size=ngram_size)
                                    for ngram_size in ngram_filter_sizes]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)
        self._output_dim = num_filters * len(ngram_filter_sizes)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def forward(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1).float()

        # The convolution layers expect input of shape (batch size, in_channels(input_dim), sequence lengths)
        inputs = torch.transpose(inputs, 1, 2)
        # Each convolutiona layer returns output of size (batch size, num of filters, pool length),
        # where pool length = sequence lengths - ngram_size + 1
        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            filter_outputs.append(F.relu(convolution_layer(inputs)).max(dim=2)[0])

        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        return maxpool_output