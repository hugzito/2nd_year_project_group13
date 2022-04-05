import gzip
import json
import torch 
import torch.nn as nn
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import torch.optim as optim
import gensim.models

def build_vocab(filepath, padding = False):
    train_vocab = {}
    train = gzip.open(filepath)
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter = 0
    if padding: 
        train_vocab['<PAD>'] = 0
        counter2 += 1
    no_reviewText = []
    labels = {}
    sentences = {}
    tokenizer = TweetTokenizer()
    for line in train:
        counter1 +=1
        #print(line)
        if 'reviewText' in json.loads(line).keys():
            a = json.loads(line)
            sentences[counter3] = a['reviewText']
            counter3 += 1
            if a['sentiment'] == 'positive':
                labels[counter] = 1
            elif a['sentiment'] == 'negative': 
                labels[counter] = 0
            counter +=1
            for word in tokenizer.tokenize(json.loads(line)['reviewText']):
                if word not in train_vocab.keys():
                    train_vocab[word] = counter2
                    counter2 += 1
        else:
            no_reviewText.append(counter1)
    final_dict = {'line_count' : counter1,
                 'review_count' : counter3,
                 'vocab_size' : counter2,
                 'no_text_reviews' : no_reviewText,
                 'labels' : labels,
                 'vocabulary' : train_vocab,
                 'sentences' : sentences}
    return final_dict

def sen_vectorizer(filepath, cutoff = False): 
    vocab, index = {}, 1
    data = gzip.open(filepath)
    vocab['<PAD>'] = 0
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter = 0
    no_reviewText = []
    sentences = {}
    tokenizer = TweetTokenizer()
    labels = {}
    for line in data:
        counter1 +=1
        #print(line)
        if 'reviewText' in json.loads(line).keys():
            a = json.loads(line)
            b = tokenizer.tokenize(a['reviewText'])
            if cutoff: 
                b = b[:cutoff]
            sentences[counter3] = b
            counter3 += 1
            if a['sentiment'] == 'positive':
                labels[counter] = 1
            elif a['sentiment'] == 'negative': 
                labels[counter] = 0
            counter +=1
            for word in b:
                if word not in vocab.keys():
                    vocab[word] = index
                    index += 1
        else:
            no_reviewText.append(counter1)
    inverse_vocab = {index: token for token, index in vocab.items()}
    final_dict = {'line_count' : counter1,
                 'review_count' : counter3,
                 'vocab_size' : counter2,
                 'no_text_reviews' : no_reviewText,
                 'labels' : labels,
                 'vocabulary' : vocab,
                 'sentences' : sentences,
                 'inverse_vocab' : inverse_vocab}
    return final_dict
    
def create_onehot(vocab, sentences, tokenzier):
    # Create matrix
    m1 = torch.zeros(len(sentences), len(vocab))
    # Correct indices
    for sen in range(len(sentences)): 
        for word in sentences[sen]: 
            if word in vocab.keys():
                m1[sen, vocab[word]] = 1
    return m1

def create_batches(matrix, batch_size,labels): 
    num_batches = int(len(matrix)/batch_size)
    feats_batches = matrix[:batch_size*num_batches].view(num_batches,batch_size, matrix.shape[1])
    bingus = labels
    num_batches = int(len(bingus)/batch_size)
    label_batches = bingus[:batch_size*num_batches].view(num_batches,batch_size,1)
    return feats_batches, label_batches

paths = {'train':'../classification/music_reviews_train.json.gz',
        'test':'../classification/music_reviews_test_masked.json.gz',
        'dev' : '../classification/music_reviews_dev.json.gz'}
embed_dim = 64

# setting up tokenizer, train and dev data
tokenizer = TweetTokenizer()
train_data = sen_vectorizer(paths['train'], cutoff = 100)
train_matrix = create_onehot(train_data['vocabulary'], train_data['sentences'], TweetTokenizer)

def rob_skipgram(tokenized_sents, tokenizer, word2idx, window_size):
    PAD = '<PAD>'
    fullData = []
    labels = []
    for sent in tokenized_sents:
        for tgtIdx in range(len(tokenized_sents[sent])):
            labels.append(word2idx[tokenized_sents[sent][tgtIdx]])
            dataLine = []
            # backwards
            for dist in reversed(range(1,window_size+1)):
                srcIdx = tgtIdx - dist
                if srcIdx < 0:
                    dataLine.append(word2idx[PAD])
                else:
                    dataLine.append(word2idx[tokenized_sents[sent][srcIdx]])
            # forwards
            for dist in range(1,window_size+1):
                srcIdx = tgtIdx + dist
                if srcIdx >= len(tokenized_sents[sent]):
                    dataLine.append(word2idx[PAD])
                else:
                    dataLine.append(word2idx[tokenized_sents[sent][srcIdx]])
            fullData.append(dataLine)
    return fullData, labels

data, labels = rob_skipgram(train_data['sentences'], tokenizer, train_data['vocabulary'], 2)
labels = torch.tensor(labels)
data = torch.tensor(data)

class CBOW(nn.Module):
    def __init__(self, emb_dim, vocab_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_dim, emb_dim)
        # note that embeddingsbag can also be used, then sum can be skipped in forward()
        self.linear = nn.Linear(emb_dim, vocab_dim)
        #self.activation_function = nn.Softmax(dim=0)
        self.loss_function = nn.CrossEntropyLoss()

    
    def forward(self, inputs, gold):
        embeds = self.embeddings(inputs)
        out = torch.sum(embeds,dim=0)
        out = self.linear(out)
        out = self.loss_function(out, gold)
        return out


cbow = CBOW(embed_dim,len(train_data['vocabulary']))

feat_batches, label_batches = create_batches(data, 1000, labels)

# compile and train the model
optimizer = optim.SGD(cbow.parameters(), lr=0.001)
counter = 0
loop_nr = 1

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    print(epoch)
    for window, label in zip(feat_batches, label_batches):
        for sub_w, sub_l in zip(window, label):
            if counter % 1000 == 0:
                print(counter, running_loss)
            window = sub_w.view(-1, 1)
            label = sub_l.view(1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = cbow.forward(window, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            counter+=1

print('Finished Training')

embeds = cbow.state_dict()['embeddings.weight']
outFile = open('embeds.txt', 'w')
outFile.write(str(len(train_data['inverse_vocab'])) + ' ' + str(embed_dim) + '\n')
for word, embed in zip(train_data['inverse_vocab'], embeds):
    outFile.write(word + ' ' + ' '.join([str(x) for x in embed.tolist()]) + '\n')
outFile.close()

tinyEmbs = gensim.models.KeyedVectors.load_word2vec_format('embeds.txt')
