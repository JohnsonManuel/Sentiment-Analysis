from LSTM_functions import SentimentLSTM, process_sequence, pad_sequences_manual

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


df = pd.read_csv('/3-ML_training/training_data.csv')
all_words = [ w for s in df.cleaned_text.values.tolist() for w in s.split()]
words = Counter(all_words)

words = {k: v for k, v in words.items() if v > 1}
# Sorting on the basis of most common words
words = sorted(words, key=words.get, reverse=True)
words = ['_PAD', '_UNK'] + words

# Creating a dict
word_to_idx = {w: i for i, w in enumerate(words)}

X = [process_sequence(s.split(), word_to_idx) for s in df.cleaned_text.values.tolist()]
y = df.target.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

MAX_LEN = max([len(s) for s in  X_train])

X_train = pad_sequences_manual(X_train, maxlen = MAX_LEN, padding = 'pre')
X_test = pad_sequences_manual(X_test, maxlen = MAX_LEN, padding = 'pre')
y_train = np.array(y_train)
y_test = np.array(y_test)

train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

batch_size = 512

train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size, drop_last = True)
test_loader = DataLoader(test_data, shuffle = True, batch_size = batch_size, drop_last = True)

is_cuda = torch.cuda.is_available()

if is_cuda:
  device = torch.device("cuda")
  print("GPU is available")
else:
  device = torch.device("cpu")
  print("GPU not available, CPU used")

vocab_size = len(word_to_idx) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 512
n_layers = 2

model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.to(device)

criterion = nn.BCELoss()
# Until here same as the training file code

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)
y_pred = []
y_true = []

model.load_state_dict(torch.load('/4-Testing_and_Evaluation/Saved models/best_LSTM_model.pt'))
model.eval()
print(model)


def process_custom(text):
    sequence = process_sequence(text.split(), word_to_idx)
    sequence = [0] * (MAX_LEN - len(sequence)) + sequence
    return torch.unsqueeze(torch.from_numpy(np.array(sequence)), 0)


def make_prediction(review, show_tensfor_shape=False):
    label_dict = {0: "negative", 1: "positive"}

    custom_review_tensor = process_custom(review)
    test_h = model.init_hidden(1)
    out, test_h = model(custom_review_tensor.to(device), test_h)

    if show_tensfor_shape == True:
        print(custom_review_tensor)

    rounded_pred = torch.round(out)

    print("Exact calculation:", out)
    print("\nPrediction: {}. Representing a {} review.".format(int(rounded_pred), label_dict[int(rounded_pred)]))

negative_review = 'This was nice but there were no charging slots and config took some time, i should have seen the mails'
make_prediction(negative_review)


model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # rounds the output to 0/1´
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

    y_pred.extend(pred.tolist())
    y_true.extend(labels.tolist())

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc * 100))


print('Classification Report:')
print(classification_report(y_true, y_pred, labels=[1,0], digits=4))