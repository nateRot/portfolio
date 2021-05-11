# https://www.kaggle.com/arunmohan003/sentiment-analysis-using-lstm-pytorch

# %%
# Setup
import os
import sys
import time
import torch
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

plt.rcParams['font.size'] = 20
data_dir = os.path.expanduser('~/.pytorch-datasets')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (f'Device used is: {device}')

# %%
import torchtext.data

# torchtext Field objects parse text (e.g. a review) and create a tensor representation

# This Field object will be used for tokenizing the movie reviews text
review_parser = torchtext.legacy.data.Field(
    sequential=True, use_vocab=True, lower=True,
    init_token='<sos>', eos_token='<eos>', dtype=torch.long,
    tokenize='spacy', tokenizer_language='en_core_web_sm'
)

# This Field object converts the text labels into numeric values (0,1,2)
label_parser = torchtext.legacy.data.Field(
    is_target=True, sequential=False, unk_token=None, use_vocab=True
)


# %%
import torchtext.datasets

# Load SST, tokenize the samples and labels
# ds_X are Dataset objects which will use the parsers to return tensors
ds_train, ds_valid, ds_test = torchtext.legacy.datasets.SST.splits(
    review_parser, label_parser, root=data_dir
)

n_train = len(ds_train)
print(f'Number of training samples: {n_train}')
print(f'Number of test     samples: {len(ds_test)}')



# %%
review_parser.build_vocab(ds_train)
label_parser.build_vocab(ds_train)

print(f"Number of tokens in training samples: {len(review_parser.vocab)}")
print(f"Number of tokens in training labels: {len(label_parser.vocab)}")


# %%
BATCH_SIZE = 4

# BucketIterator creates batches with samples of similar length
# to minimize the number of <pad> tokens in the batch.
dl_train, dl_valid, dl_test = torchtext.legacy.data.BucketIterator.splits(
    (ds_train, ds_valid, ds_test), batch_size=BATCH_SIZE,
    shuffle=True, device=device)


# %%
batch = next(iter(dl_train))

X, y = batch.text.to(device), batch.label.to(device)
# X = X.unsqueeze(-1)
print(f'X = \n {X} {X.shape} \n\n')
print(f'y = \n {y} {y.shape} \n\n')



# %%
# loading weights from glove 
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_input_file="/home/nate/Technion/236781_Deep_Learning_on_Computational_Accelerators/Assignments/assignment_4/project/glove.6B.50d.txt", word2vec_output_file="emb_word2vec_format.txt")

import gensim
import torch
model = gensim.models.KeyedVectors.load_word2vec_format('emb_word2vec_format.txt')
weights = torch.FloatTensor(model.vectors)

# %%
import torch.nn as nn
class SentimentRNN(nn.Module):
    def __init__(self, vocab_dim, weights, h_dim, out_dim):
        super().__init__()
        
        # nn.Embedding loading from pretrained weights
        self.embedding = nn.Embedding.from_pretrained(weights)
        
        # Our own Vanilla RNN layer, without phi_y so it outputs a class score
        self.rnn = nn.RNN(input_size=weights.shape[1], hidden_size=h_dim)
        
        self.fc = nn.Linear(h_dim, out_dim)

        # To convert class scores to log-probability we'll apply log-softmax
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, X):
        # X shape: (S, B) Note batch dim is not first!
        
        embedded = self.embedding(X) # embedded shape: (S, B, E)
        
        # Loop over (batch of) tokens in the sentence(s)
        ht = None
        for xt in embedded:           # xt is (B, E)
            yt, ht = self.rnn(xt.unsqueeze(0), ht) # yt is (B, D_out)
        
        # Push through fc layer
        yt = self.fc(yt)
        
        # Class scores to log-probability
        yt_log_proba = self.log_softmax(yt).squeeze()
        
        return yt_log_proba



# %%
INPUT_DIM = len(review_parser.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 3

model = SentimentRNN(INPUT_DIM, weights, HIDDEN_DIM, OUTPUT_DIM).to(device)
model


# %%
print(f'model(X) = \n', model(X), model(X).shape)
print(f'labels = ', y)



# %%
def train(model, optimizer, loss_fn, dataloader, max_epochs=100, max_batches=200):
    for epoch_idx in range(max_epochs):
        total_loss, num_correct = 0, 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            X, y = batch.text, batch.label
            # X = X.unsqueeze(-1)
            # Forward pass
            y_pred_log_proba = model(X).squeeze()

            # Backward pass
            optimizer.zero_grad()
            loss = loss_fn(y_pred_log_proba, y)
            loss.backward()

            # Weight updates
            optimizer.step()

            # Calculate accuracy
            total_loss += loss.item()
            y_pred = torch.argmax(y_pred_log_proba, dim=1)
            num_correct += torch.sum(y_pred == y).float().item()

            if batch_idx == max_batches-1:
                break
                
        print(f"Epoch #{epoch_idx}, loss={total_loss /(max_batches):.3f}, accuracy={num_correct /(max_batches*BATCH_SIZE):.3f}, elapsed={time.time()-start_time:.1f} sec")



# %%
import torch.optim as optim

rnn_model = SentimentRNN(INPUT_DIM, weights, HIDDEN_DIM, OUTPUT_DIM).to(device)

optimizer = optim.Adam(rnn_model.parameters(), lr=1e-3)

# Recall: LogSoftmax + NLL is equiv to CrossEntropy on the class scores
loss_fn = nn.NLLLoss()

train(rnn_model, optimizer, loss_fn, dl_train, max_epochs=100) # just a demo

print('End reached')