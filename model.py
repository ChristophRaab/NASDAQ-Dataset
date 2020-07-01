import torch, torch.nn as nn
# from functions import ReverseLayerF
from torch.nn import functional as F
from torch.nn.functional import embedding
from torch.utils.data import TensorDataset
import numpy as np

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class LinearModel(nn.Module):

    def __init__(self, embedding_matrix_sg, embedding_matrix_gl, number_class, drop=0.25):
        super(LinearModel, self).__init__()

        # Skipgram Embedding
        self.embedding_sg, num_embeddings_sg, embedding_dim_sg = create_emb_layer(embedding_matrix_sg, True)

        # Glove Embedding
        self.embedding_gl, num_embeddings_gl, embedding_dim_gl = create_emb_layer(embedding_matrix_gl, True)

        # Trained Embedding
        self.embedding_tr, num_embeddings_tr, embedding_dim_tr = create_emb_layer(np.random.rand(num_embeddings_gl, embedding_dim_gl), False)

        # Densenet121
        self.d121 = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        self.avgpool = nn.AvgPool2d(2)
        self.dense = nn.Linear(1024, number_class)
        self.softmax = nn.Softmax(dim=1)

        # Define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, rho=0.9, momentum=0.0, epsilon=1e-07) #RMSprop not found?
        self.optimizer = torch.optim.Adam(self.parameters())

        # feature encoder1
        #self.f1 = nn.Linear(input_dim1, 100)
        #self.f1_drop = nn.Dropout(drop)

        # feature encoder2
        #self.g1 = nn.Linear(input_dim2, 100)
        #self.g1_drop = nn.Dropout(drop)

        # decoder
        #self.d1 = nn.Linear(100, 100)
        #self.d2 = nn.Linear(100, input_dim2)
        #self.d_drop = nn.Dropout(drop)

        # sentiment classifier
        # self.sc1 = nn.Linear(200, 10)
        # self.sc2 = nn.Linear(10, 2)
        # self.sc_drop = nn.Dropout(drop)

        # domain classifier
        #self.dc1 = nn.Linear(200, 10)
        #self.dc2 = nn.Linear(10, 2)
        #self.dc_drop = nn.Dropout(drop)

    # def encode1(self, x1):
    #     x1 = self.f1(x1)
    #     x1 = F.relu(x1)
    #     x1 = self.f1_drop(x1)
    #     return x1
    #
    # def encode2(self, x2):
    #     x2 = self.g1(x2)
    #     x2 = F.relu(x2)
    #     x2 = self.g1_drop(x2)
    #     return x2
    #
    # def decode(self, z):
    #     z = self.d1(z)
    #     z = F.relu(z)
    #     z = self.d_drop(z)
    #     # z = torch.sigmoid(self.d2(z))
    #     z = self.d2(z)
    #     return z
    #
    # def domain_classifier(self, h):
    #     h = self.dc1(h)
    #     # h = self.dc_drop(h)
    #     h = F.relu(h)
    #     h = self.dc2(h)
    #     return h
    #
    # def sentiment_classifier(self, h):
    #     h = self.sc1(h)
    #     h = self.sc_drop(h)
    #     h = F.relu(h)
    #     h = self.sc2(h)
    #     return h

    def forward(self, input, alpha):
        e1 = self.embedding_sg(input)
        e2 = self.embedding_gl(input)
        e3 = self.embedding_tr(input)

        emd_stack = torch.stack([e1, e2, e3])

        self.d121.eval()
        d121_out = self.d121(emd_stack)
        out = self.avgpool(d121_out)
        out = self.dense(out)
        out = self.softmax(out)

        #reverse_z = ReverseLayerF.apply(z, alpha)
        #class_output = self.sentiment_classifier(z)
        #domain_output = self.domain_classifier(reverse_z)

        #return reconstructed, class_output, domain_output
        return out


    def train_model(self, trainloader, epochs=10):
        self.train()

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')