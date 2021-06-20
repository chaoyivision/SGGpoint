'''
SubNetworks used for SGGpoint (Chaoyi Zhang), including:
    1. backbone networks (PointNet & DGCNN);
    2. MLP-tails (NodeMLP & EdgeMLP);
    3. edge feat. initialization func.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SyncBatchNorm

#####################################################
#                                                   #
#                                                   #
#   Backbone network - PointNet                     #
#                                                   #
#                                                   #
#####################################################

class PointNet(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel, embeddings):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embeddings, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embeddings)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        return x


#####################################################
#                                                   #
#                                                   #
#   Backbone network - DGCNN (and its components)   #
#                                                   #
#                                                   #
#####################################################

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class DGCNN(nn.Module):
    # official DGCNN
    def __init__(self, input_channel, embeddings):
        super(DGCNN, self).__init__()
        self.k = 20
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel * 2, 64, kernel_size=1, bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, embeddings, kernel_size=1, bias=False),nn.BatchNorm1d(embeddings),nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        #x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv5(x)
        return x

###############################################
#                                             #
#                                             #
#   Tail Classification - NodeMLP & EdgeMLP   #
#                                             #
#                                             #
###############################################

class NodeMLP(nn.Module):
    def __init__(self, embeddings, nObjClasses, negative_slope=0.2):
        super(NodeMLP, self).__init__()
        mid_channels = embeddings // 2
        self.node_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
        self.node_BnReluDp = nn.Sequential(SyncBatchNorm(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
        self.node_linear2 = nn.Linear(mid_channels, nObjClasses, bias=False)

    def forward(self, node_feats):
        # node_feats: (1, nodes, embeddings)  => node_logits: (1, nodes, nObjClasses)
        x = self.node_linear1(node_feats)
        x = self.node_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
        node_logits = self.node_linear2(x)
        return node_logits

class EdgeMLP(nn.Module):
    def __init__(self, embeddings, nRelClasses, negative_slope=0.2):
        super(EdgeMLP, self).__init__()
        mid_channels = embeddings // 2
        self.edge_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
        self.edge_BnReluDp = nn.Sequential(SyncBatchNorm(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
        self.edge_linear2 = nn.Linear(mid_channels, nRelClasses, bias=False)

    def forward(self, edge_feats):
        # edge_feats: (1, edges, embeddings)  => edge_logits: (1, edges, nRelClasses)
        x = self.edge_linear1(edge_feats)
        x = self.edge_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
        edge_logits = self.edge_linear2(x)
        return edge_logits

#####################################################
#                                                   #
#                                                   #
#   Edge features initialization                    #
#                                                   #
#                                                   #
#####################################################

def edge_feats_initialization(node_feats, batchwise_edge_index):
    node_feats = node_feats.squeeze(0)

    connections_from_subject_to_object = batchwise_edge_index.t()
    subject_idx = connections_from_subject_to_object[:, 0]
    object_idx = connections_from_subject_to_object[:, 1]

    subject_feats = node_feats[subject_idx]
    object_feats = node_feats[object_idx]
    diff_feats = object_feats - subject_feats

    edge_feats = torch.cat((subject_feats, diff_feats), dim=1)  # equivalent to EdgeConv (with in DGCNN)

    return edge_feats  # (num_Edges, Embeddings * 2)