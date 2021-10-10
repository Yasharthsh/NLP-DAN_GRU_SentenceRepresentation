'''
author: Sounak Mondal
'''

# std lib imports
from typing import Dict

# external libs
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

class SequenceToVector(nn.Module):
    """
    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``torch.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``torch.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : torch.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : torch.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2, device = 'cpu'):
        super(DanSequenceToVector, self).__init__(input_dim)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.relu = nn.ReLU()
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        
        # do masking with the input vector squence to get all the real tokens
        sequence_mask = torch.reshape(sequence_mask,(vector_sequence.shape[0],vector_sequence.shape[1],1))
        num_length = torch.sum(sequence_mask, axis=1)
        vector_sequence = sequence_mask * vector_sequence

        if training:
            # implement dropout probability matrix and do product of it with vector_sequence
            dropout_matrix = (torch.rand(size=(vector_sequence.shape[0],vector_sequence.shape[1], 1)) >= self.dropout).type(torch.float32)
            num_length = torch.sum(dropout_matrix * sequence_mask, axis=1)
            vector_sequence = dropout_matrix * vector_sequence
        # do an averaging of the word vectors after doing sum of these word vector along each row
        combined_vector = torch.div(torch.sum(vector_sequence, axis=1), num_length)
        # for each layer do a relu activation except for the last one
        nnLayer = []
        for i in range(0,self.num_layers):
            if i!=(self.num_layers-1):
                nnLayer.append(F.relu(self.linear_1(combined_vector)))
        # doing only linear for the last layer below
        combined_vector = self.linear_1(combined_vector)
        nnLayer.append(combined_vector)
        layer_representations = torch.stack(nnLayer, axis=1)

        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int, device = 'cpu'):
        super(GruSequenceToVector, self).__init__(input_dim)
        
        # create a torch GRU layer which takes number of dimensions and number of layers
        self.nnGRUmodel = nn.GRU(input_dim,input_dim,num_layers)
        self.num_layers = num_layers


    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:

        # do masking with the input vector squence to get all the real tokens
        sequence_mask = torch.reshape(sequence_mask,(vector_sequence.shape[0],vector_sequence.shape[1],1))
        # calculate the actual length of mini batches to do dynamic batch size
        num_words = torch.sum(sequence_mask, axis=1)
        # pass this mini batches to packed sequence to create data tensor and length tensor (which consists of variable length of that particular batch) 
        packed_embedded = nn.utils.rnn.pack_padded_sequence(vector_sequence,lengths = num_words.squeeze(), batch_first = True,enforce_sorted = False)
        combined_vector, layer_representations = self.nnGRUmodel(packed_embedded)
        combined_vector = layer_representations[-1]
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
