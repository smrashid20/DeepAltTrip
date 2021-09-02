import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import data_generator
import parameters

np.random.seed(1234567890)
torch.manual_seed(1234567890)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_graph(adj, add_eye=True):
    for i in range(len(adj)):
        adj[i][i] = 0.0

    if add_eye:
        adj_ = adj + np.eye(adj.shape[0])
    else:
        adj_ = adj
    rowsum = np.array(adj_.sum(1))
    deg_mat = np.diag(rowsum)
    degree_mat_inv_sqrt = np.sqrt(np.linalg.inv(deg_mat))
    adj_normalized = np.matmul(np.matmul(degree_mat_inv_sqrt, adj_), degree_mat_inv_sqrt)
    return adj_normalized


def param_init(input_dim, output_dim):
    initial = torch.rand(input_dim, output_dim)
    return nn.Parameter(initial)


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = param_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.matmul(Z, Z.t())
    return A_pred


class GAE(nn.Module):
    def __init__(self, adj):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(len(data_generator.vocab_to_int) - 3, parameters.ae_hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(parameters.ae_hidden1_dim, parameters.ae_hidden2_dim, adj,
                                        activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return Z, A_pred


def train_GAE_rmsLoss(adj_matrix, add_eye=True):
    adj_train = adj_matrix
    adj = adj_train

    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    features = np.eye(num_nodes, dtype=np.float)

    if add_eye:
        adj_label = adj_train + np.eye(num_nodes)
    else:
        adj_label = adj_train

    adj_norm = torch.FloatTensor(adj_norm).to(device)
    adj_label = torch.FloatTensor(adj_label).to(device)
    features = torch.FloatTensor(features).to(device)

    model = GAE(adj_norm).to(device)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=parameters.ae_learning_rate)

    def get_rms_loss(adj_rec, adj_label_):
        rms_loss = torch.sqrt(torch.sum((adj_label_ - adj_rec) ** 2))
        return rms_loss.item()

    for epoch in range(parameters.ae_num_epoch):
        t = time.time()
        optimizer.zero_grad()

        _, A_pred = model(features)
        loss = torch.sum((A_pred - adj_label) ** 2)
        loss.backward()
        optimizer.step()

        train_rms_loss = get_rms_loss(A_pred, adj_label)

        if epoch % 200 == 0 or epoch == parameters.ae_num_epoch - 1:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                  "train_rms_loss = ", "{:.5f}".format(train_rms_loss),
                  "time=", "{:.5f}".format(time.time() - t))

    print("\n\n")

    Z_final, A_pred_final = model(features)

    return np.array(Z_final.cpu().detach().numpy())


def train_GAE_BCELoss(adj_matrix):
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    adj_train = adj_matrix
    adj = adj_train

    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    features = np.eye(num_nodes, dtype=np.float)
    adj_label = adj_train

    adj_norm = torch.FloatTensor(adj_norm).to(device)
    adj_label = torch.FloatTensor(adj_label).to(device)
    features = torch.FloatTensor(features).to(device)

    model = GAE(adj_norm).to(device)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=parameters.ae_learning_rate)

    for epoch in range(parameters.ae_num_epoch):
        optimizer.zero_grad()

        _, A_pred = model(features)
        loss = F.binary_cross_entropy_with_logits(A_pred.view(-1), adj_label.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == parameters.ae_num_epoch - 1:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()))

    print("\n\n")
    Z_final, A_pred_final = model(features)

    return np.array(Z_final.cpu().detach().numpy())


def get_POI_embeddings(load_from_file=False):
    if not load_from_file:
        parameters.ae_hidden1_dim = 20
        parameters.ae_hidden2_dim = 16
        parameters.ae_num_epoch = 5000
        parameters.ae_learning_rate = 0.05

        Z_final_cat = train_GAE_BCELoss(data_generator.poi_category_matrix)
        Z_final_norm_cat = np.linalg.norm(Z_final_cat, axis=1, keepdims=True)
        Z_final_cat = Z_final_cat / Z_final_norm_cat

        # cat_r = np.matmul(Z_final_cat,Z_final_cat.T)
        # plt.imshow(cat_r, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        parameters.ae_hidden1_dim = 32
        parameters.ae_hidden2_dim = 24
        parameters.ae_num_epoch = 20000
        parameters.ae_learning_rate = 0.01

        Z_final_dist = train_GAE_rmsLoss(data_generator.poi_distance_matrix)

        # dist_r = np.matmul(Z_final_dist, Z_final_dist.T)
        # plt.imshow(dist_r, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        # parameters.ae_hidden1_dim = 32
        # parameters.ae_hidden2_dim = 24
        # parameters.ae_num_epoch = 20000
        # parameters.ae_learning_rate = 0.01
        #
        # Z_final_trans = train_GAE_rmsLoss(adj_matrix=data_generator.poi_transition_matrix_normalized,
        #                                   add_eye=False)

        # trans_r = np.matmul(Z_final_trans, Z_final_trans.T)
        # plt.imshow(trans_r, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        Z_final_concat = np.concatenate([Z_final_dist, Z_final_cat], axis=1)

        np.save(os.path.join("model_files", "POI_embedding_" + data_generator.embedding_name + ".npy"),
                Z_final_concat)

    Zb = np.load(os.path.join("model_files", "POI_embedding_" + data_generator.embedding_name + ".npy"))
    return Zb