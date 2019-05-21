import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F


# print(torch.tensor([[1],[2]]).size(0))

def normalize_size(y_pred, y_truth):
    """
    Normalize tensor size
    :param y_pred: the output of the model, if value is 3d tensor, resize to 2d
    :param y_truth: target predictions, if value is matrix, resize to vector
    :return:
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.view(-1, y_pred.size(2))

    if len(y_truth.size()) == 2:
        y_truth = y_truth.view(-1)

    return y_pred, y_truth


def sequence_loss(y_pred, y_truth, mask_index):
    """
    Compute cross entropy loss of 3d tensor with 2d tensor
    :param y_pred:
    :param y_truth:
    :param mask_index:
    :return:
    """
    y_pred, y_truth = normalize_size(y_pred, y_truth)
    return F.cross_entropy(y_pred, y_truth, ignore_index=mask_index)


def indices_to_string(indices, char_vocab, print_start=True):
    """
    Convert list idx to string
    :param indices:
    :param char_vocab:
    :param print_start:
    :return:
    """
    out_string = ""
    for idx in indices:
        if not print_start and idx == char_vocab.begin_seq_index:
            continue
        elif idx == char_vocab.mask_index or idx == char_vocab.end_seq_index:
            return out_string
        else:
            out_string += char_vocab.lookup_index(idx)

    return out_string


def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_size(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


def sample_from_model(model, vectorizer, num_samples=1, sample_size=20, temperature=1.0, national_index=None):
    """
    Get sample results, a sequence of indices from the model
    :param model: generate model that trained
    :param vectorizer:
    :param num_samples: number of sample
    :param sample_size: max length of a sample
    :param temperature: change distribute probability, < 0 make more different, > 1 make more uniform
    :param national_index: int, index of national for init hidden state
    :return:
    """
    begin_seq_index_vec = [vectorizer.char_vocab.begin_seq_index for _ in range(num_samples)]
    begin_seq_index_vec = torch.tensor(begin_seq_index_vec, dtype=torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index_vec]
    if not national_index:
        h_t = None
    else:
        h_t = model.nation_embed(torch.tensor(national_index)).expand(1, num_samples,-1)
    for tim_step in range(sample_size):
        x_t = indices[tim_step]
        x_embed_t = model.char_embed(x_t)
        rnn_out_t, h_t = model.rnn(x_embed_t, h_t)
        predict_vector = model.fc(rnn_out_t.squeeze(dim=1))
        probability_vector = torch.softmax(predict_vector / temperature, dim=1)
        indices.append(torch.multinomial(probability_vector, num_samples=1))
    indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices

#
# begin_seq_index = [0 for _ in range(4)]
# begin_seq_index = torch.tensor(begin_seq_index, dtype=torch.int64).unsqueeze(dim=0)
# print([begin_seq_index])
