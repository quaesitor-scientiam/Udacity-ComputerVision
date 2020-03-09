import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        # super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # turns words into a vector of specified size aka embedded word vector
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # LSTM takes embedded word vectors (of specified size) as inputs
        # and outputs hidden states of size hidden_size
        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size)

        # linear layer maps the hidden state output dimension
        # to the number of tags we want as output, num_layers
        # making this a fully connected layer
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        # activations
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, features, captions):
        hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()

        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()

        # create embedded word vectors for each word in a sentence
        embeds = self.word_embeddings(captions)

        for w in range(captions.size(1)):
            if w == 0:  #first word
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
            else:  # for the rest
                hidden_state, cell_state = self.lstm_cell(embeds[:, w, :], (hidden_state, cell_state))
        
            # output of attention mechanism
            out = self.fc_out(hidden_state)

            # construct output tensor
            outputs[:, w, :] = out
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        sampled_ids = []

        hx = None
        features = inputs.view(1,512)
        for t in range(max_len):
            hiddens, states = self.lstm_cell(features, hx)
            outputs = self.fc_out(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            features = self.word_embeddings(predicted).unsqueeze(1)
            hx = (hiddens, states)

        ids = torch.stack(sampled_ids, 1)
        result = ids.view(max_len,1).squeeze().tolist()
        return result
    
