from LSTM_functions import process_sequence, pad_sequences_manual, SentimentLSTM
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from collections import Counter
from sklearn.model_selection import train_test_split



df = pd.read_csv('training_data.csv')
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
print(model)

lr = 0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2
counter = 0
print_every = 50
clip = 5
test_loss_min = np.Inf

model.train()  # model set to training mode


for i in range(epochs):
  h = model.init_hidden(batch_size)

  for inputs, labels in train_loader:
    counter += 1
    inputs, labels = inputs.to(device), labels.to(device)
    h = tuple([e.data for e in h])
    model.zero_grad()  # reset gradients
    output, h = model(inputs, h)  # forward propagation
    loss = criterion(output.squeeze(), labels.float())  # compute loss
    loss.backward()  # backward propagation
    nn.utils.clip_grad_norm_(model.parameters(), clip)  # regularization by clipping gradient
    optimizer.step()  # apply gradients

    # METRICS:
    if counter % print_every == 0:
      test_h = model.init_hidden(batch_size)
      test_losses = []
      num_correct = 0
      model.eval()
      for inp, lab in test_loader:
        test_h = tuple([each.data for each in test_h])
        inp, lab = inp.to(device), lab.to(device)
        out, test_h = model(inp, test_h)
        test_loss = criterion(out.squeeze(), lab.float())
        test_losses.append(test_loss.item())

        # Compute acc
        pred = torch.round(out.squeeze())  # rounds the output to 0/1
        correct_tensor = pred.eq(lab.float().view_as(pred))  # calculate how many preds == labels
        correct = np.squeeze(correct_tensor.cpu().numpy())  # like np.ravel
        num_correct += np.sum(correct)  # sums matches

      print("Test loss: {:.3f}".format(np.mean(test_losses)))
      test_acc = num_correct / len(test_loader.dataset)
      print("Test accuracy: {:.3f}%".format(test_acc * 100))

      model.train()
      print("Epoch: {}/{}...".format(i + 1, epochs),
            "Step: {}...".format(counter),
            "Loss: {:.6f}...".format(loss.item()),
            "Test Loss: {:.6f}".format(np.mean(test_losses)))
      if np.mean(test_losses) <= test_loss_min:
        torch.save(model.state_dict(), '/4-Testing_and_Evaluation/Saved models/best_LSTM_model.pt')
        print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min, np.mean(test_losses)))
        test_loss_min = np.mean(test_losses)