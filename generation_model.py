import torch
import torch.nn as nn
import numpy as np


class SurnameGenerationModel(nn.Module):
    """
    Generation model generate surname without caring about national info
    """
    def __init__(self, char_embedding_size, char_vocab_size, rnn_hidden_size,
                 batch_first=True, padding_idx=0, dropout_p=0.5):
        """
        Args
        :param char_embedding_size:
        :param char_vocab_size:
        :param rnn_hidden_size:
        :param batch_first:
        :param padding_idx:
        :param dropout_p:
        """

        super(SurnameGenerationModel, self).__init__()

        self.char_embed = nn.Embedding(num_embeddings=char_vocab_size,
                                       embedding_dim=char_embedding_size,
                                       padding_idx=padding_idx)

        self.rnn = nn.GRU(input_size=char_embedding_size, hidden_size=rnn_hidden_size, batch_first=batch_first)
        self.fc = nn.Linear(in_features=rnn_hidden_size, out_features=char_vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self._char_vocab_size = char_vocab_size

    def forward(self, x_in, apply_softmax=False):
        x_embedded = self.char_embed(x_in)

        y_out, final_hidden_state = self.rnn(x_embedded)

        batch_size, seq_size, feature_size = y_out.shape

        y_out = y_out.contiguous().view(batch_size * seq_size, feature_size)

        y_out = self.fc(self.dropout(y_out))

        if apply_softmax:
            y_out = torch.softmax(y_out, dim=1)

        y_out = y_out.view(batch_size, seq_size, self._char_vocab_size)

        return y_out


class SurnameGenerationWithConditionModel(nn.Module):
    """
    Generation model generate surname that care about national info
    """
    def __init__(self, char_embedding_size, char_vocab_size, rnn_hidden_size,
                 num_nationalities,
                 batch_first=True, padding_idx=0, dropout_p=0.5):
        """
        Args
        :param char_embedding_size:
        :param char_vocab_size:
        :param rnn_hidden_size:
        :param batch_first:
        :param padding_idx:
        :param dropout_p:
        """

        super(SurnameGenerationWithConditionModel, self).__init__()

        self.char_embed = nn.Embedding(num_embeddings=char_vocab_size,
                                       embedding_dim=char_embedding_size,
                                       padding_idx=padding_idx)

        self.nation_embed = nn.Embedding(num_embeddings=num_nationalities,
                                         embedding_dim=rnn_hidden_size)

        self.rnn = nn.GRU(input_size=char_embedding_size, hidden_size=rnn_hidden_size, batch_first=batch_first)
        self.fc = nn.Linear(in_features=rnn_hidden_size, out_features=char_vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self._char_vocab_size = char_vocab_size

    def forward(self, x_in, nationality_index, apply_softmax=False):
        x_embedded = self.char_embed(x_in)
        nation_embedded = self.nation_embed(nationality_index).unsqueeze(0)

        y_out, final_hidden_state = self.rnn(x_embedded, nation_embedded)

        batch_size, seq_size, feature_size = y_out.shape

        y_out = y_out.contiguous().view(batch_size * seq_size, feature_size)

        y_out = self.fc(self.dropout(y_out))

        if apply_softmax:
            y_out = torch.softmax(y_out, dim=1)

        y_out = y_out.view(batch_size, seq_size, self._char_vocab_size)

        return y_out

# print(
#     SurnameGenerationModel(char_embedding_size=20, char_vocab_size=10, rnn_hidden_size=25)(
#         torch.randint(low=0, high=10, size=(20, 35))
#     ).shape
# )

#
# print(
#     SurnameGenerationWithConditionModel(char_embedding_size=20, char_vocab_size=10, rnn_hidden_size=25, num_nationalities=3)(
#         torch.randint(low=0, high=10, size=(20, 35)), torch.randint(low=0, high=3, size=(20,))
#     ).shape
# )
