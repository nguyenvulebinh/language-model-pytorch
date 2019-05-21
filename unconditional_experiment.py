from surname_dataset import SurnameDataset
from generation_model import SurnameGenerationModel
from torch.optim import Adam
import utils
from surname_dataset import generate_batches
import torch

dataset = SurnameDataset.load_dataset_and_make_vectorizer('./data/surnames_with_splits.csv')
vectorizer = dataset.get_vectorizer()
generater = SurnameGenerationModel(char_embedding_size=128, char_vocab_size=len(vectorizer.char_vocab),
                                   rnn_hidden_size=256, padding_idx=vectorizer.char_vocab.mask_index)
optimizer = Adam(generater.parameters(), lr=0.01)

for epoch_index in range(10):
    # Train step
    generater.train()
    loss_epoch = 0
    acc_epoch = 0
    dataset.set_split('train')
    for batch_index, batch_dict in enumerate(generate_batches(dataset, batch_size=128)):
        generater.zero_grad()

        y_pred = generater(batch_dict['x_data'])

        loss = utils.sequence_loss(y_pred, batch_dict['y_target'], vectorizer.char_vocab.mask_index)
        acc = utils.compute_accuracy(y_pred, batch_dict['y_target'], vectorizer.char_vocab.mask_index)

        loss_epoch += (loss.item() - loss_epoch) / (batch_index + 1)
        acc_epoch += (acc - acc_epoch) / (batch_index + 1)

        loss.backward()

        optimizer.step()

    print("epoch {}: train loss {}, acc: {:.2f}".format(epoch_index, loss_epoch, acc_epoch))

    # Validate step
    generater.eval()
    loss_epoch = 0
    acc_epoch = 0
    dataset.set_split('val')
    for batch_index, batch_dict in enumerate(generate_batches(dataset, batch_size=128)):
        y_pred = generater(batch_dict['x_data'])

        loss = utils.sequence_loss(y_pred, batch_dict['y_target'], vectorizer.char_vocab.mask_index)
        acc = utils.compute_accuracy(y_pred, batch_dict['y_target'], vectorizer.char_vocab.mask_index)

        loss_epoch += (loss.item() - loss_epoch) / (batch_index + 1)
        acc_epoch += (acc - acc_epoch) / (batch_index + 1)

    index_view = torch.randint(low=0, high=len(batch_dict['x_data']), size=(1,)).item()
    print("{}\n{}".format(utils.indices_to_string(batch_dict['x_data'][index_view].numpy(), vectorizer.char_vocab),
                          utils.indices_to_string(torch.argmax(torch.softmax(y_pred, dim=2)[index_view], dim=1).numpy(),
                                                  vectorizer.char_vocab)))

    print("epoch {}: train loss {}, acc: {:.2f}\n\n".format(epoch_index, loss_epoch, acc_epoch))

samples_indices = utils.sample_from_model(model=generater,
                                          vectorizer=vectorizer,
                                          num_samples=10,
                                          sample_size=10,
                                          temperature=0.5)

print("Sample generate: ")
for indices in samples_indices:
    print(utils.indices_to_string(indices.numpy(), vectorizer.char_vocab, print_start=False))
