from collections import defaultdict
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags):
        super(CNNclass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Conv 1d
        self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_size,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.projection_layer = torch.nn.Linear(in_features=num_filters, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words):
        emb = self.embedding(words)                 # nwords x emb_size
        emb = emb.unsqueeze(0).permute(0, 2, 1)     # 1 x emb_size x nwords
        h = self.conv_1d(emb)                       # 1 x num_filters x nwords
        # Do max pooling
        h = h.max(dim=2)[0]                         # 1 x num_filters
        h = self.relu(h)
        out = self.projection_layer(h)              # size(out) = 1 x ntags
        return out

class CNN_Text(torch.nn.Module):

    def __init__(self, nwords, emb_size, num_filters, window_size, ntags):
        super(CNN_Text, self).__init__()

        V = nwords
        D = emb_size
        C = num_filters
        Ci = 1
        Co = window_size

        self.embed = torch.nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        # self.convs1 = torch.nn.ModuleList([torch.nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.convs1 = []
        self.convs1.append(torch.nn.Conv2d(Ci, Co, (3, D)))
        self.convs1.append(torch.nn.Conv2d(Ci, Co, (4, D)))
        self.convs1.append(torch.nn.Conv2d(Ci, Co, (5, D)))
        self.fc1 = torch.nn.Linear(in_features=num_filters, out_features=ntags, bias=True)


    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        logit = self.fc1(x)  # (N, C)
        return logit


class CNN(nn.Module):
    # def __init__(self, **kwargs):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags, w_matrix):
        super(CNN, self).__init__()

        self.BATCH_SIZE = 50
        self.MAX_SENT_LEN = window_size
        self.WORD_DIM = emb_size
        self.VOCAB_SIZE = nwords
        self.CLASS_SIZE = ntags
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM = num_filters
        self.DROPOUT_PROB = 0.5
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))
        self.WV_MATRIX = w_matrix
        self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        self.embedding.weight.requires_grad = False

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)

        return x


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


# Read in the data
train = list(read_dataset("../topicclass/topicclass_train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../topicclass/topicclass_valid.txt"))
test = list(read_dataset("../topicclass/topicclass_test.txt"))
nwords = len(w2i)
ntags = len(t2i)
print(t2i)
# Define the model
EMB_SIZE = 64
WIN_SIZE = 3
FILTER_SIZE = 64


print("loading word2vec...")
word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

wv_matrix = []
for word,tag in train:
    if word in word_vectors.vocab:
        wv_matrix.append(word_vectors.word_vec(word))
    else:
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

# one for UNK and one for zero padding
wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
wv_matrix.append(np.zeros(300).astype("float32"))
wv_matrix = np.array(wv_matrix)

# initialize the model
model = CNN(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, ntags,wv_matrix)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()


for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for words, tag in train:
        # Padding (can be done in the conv layer as well)
        if len(words) < WIN_SIZE:
            words += [0] * (WIN_SIZE - len(words))
        words_tensor = torch.tensor(words).type(type)
        tag_tensor = torch.tensor([tag]).type(type)
        scores = model(words_tensor)
        predict = scores[0].argmax().item()
        if predict == tag:
            train_correct += 1

        my_loss = criterion(scores, tag_tensor)
        train_loss += my_loss.item()
        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (
        ITER, train_loss / len(train), train_correct / len(train), time.time() - start))
    # Perform testing
    test_correct = 0.0
    for words, tag in dev:
        # Padding (can be done in the conv layer as well)
        if len(words) < WIN_SIZE:
            words += [0] * (WIN_SIZE - len(words))
        words_tensor = torch.tensor(words).type(type)
        scores = model(words_tensor)[0]
        predict = scores.argmax().item()
        print(predict)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct / len(dev)))
    for words, tag in test:
        # Padding (can be done in the conv layer as well)
        if len(words) < WIN_SIZE:
            words += [0] * (WIN_SIZE - len(words))
        words_tensor = torch.tensor(words).type(type)
        scores = model(words_tensor)[0]
        predict = scores.argmax().item()
        print(predict)