import math
import random
from typing import Dict, Optional

from src.base.model import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn.conv as Conv
from torch_geometric.typing import OptTensor

import numpy as np
import dgl
from torch.nn import Parameter
from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv

import src.mainfold

# =========================
# =========================

class HIPPIEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        args=None,
        act='relu',
        layer_num=2,
        radius=None,
        dropout=0.0,
        if_bias=True,
        use_att=0,
        local_agg=0,
        class_num=7,
        in_len=512,
        device=None,
    ):
        super(HIPPIEncoder, self).__init__()
        self.models = torch.nn.ModuleList()  # seven independent GNN models
        self.layer_num = layer_num
        self.class_num = class_num
        self.in_len = in_len
        self.input_dim = input_dim
        self.hyper_dim = int(self.input_dim / 2)
        self.manifold = src.mainfold.Hyperboloid()
        self.device = device

        dims = [self.input_dim] + ([self.hyper_dim] * (layer_num))
        if self.manifold.name == 'Hyperboloid':
            dims[0] += 1

        n_curvatures = len(dims) + 1
        self.radius = radius
        if radius is None:
            self.curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
        else:
            self.curvatures = [torch.tensor([radius]) for _ in range(n_curvatures)]  # fixed curvature

        act_fn = getattr(torch.nn.functional, act)
        acts = [act_fn] * layer_num

        for c in range(class_num):
            graph_layers = []
            i = 0
            c_in, c_out = self.curvatures[i + 1], self.curvatures[i + 2]
            in_dim, out_dim = dims[i], dims[i + 1]
            graph_layers.append(
                HyperbolicGCN(
                    self.manifold,
                    in_dim,
                    out_dim,
                    c_in,
                    c_out,
                    dropout,
                    acts[i],
                    if_bias,
                    use_att,
                    local_agg,
                )
            )
            graph_layers.append(
                HyperbolicDecoder(
                    dims[-2],
                    dims[-1],
                    if_bias,
                    dropout,
                    self.curvatures[-1],
                )
            )
            graph_layers.append(
                torch_geometric.nn.models.GIN(
                    dims[-1],
                    dims[-1],
                    1,
                    out_dim,
                    act='tanh',
                    norm=nn.BatchNorm1d(dims[-1]),
                )
            )
            self.models.append(nn.Sequential(*graph_layers))

        self.output_dim = dims[0] + 1 * class_num * dims[-1]

    def forward(self, data):
        """
        - data.embed1        : [num_nodes, input_dim]
        """
        f1 = data.embed1
        sparse_adj = data.sparse_adj1
        edges = data.edge1

        o = torch.zeros_like(f1)
        f1 = torch.cat([o[:, 0:1], f1], dim=1)

        output = [f1]

        x_tan = self.manifold.proj_tan0(f1, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        for i, m in enumerate(self.models):
            tmp = x_hyp
            tmp, _ = m[0]((tmp, sparse_adj[i]))  # HyperbolicGCN
            tmp = m[1].forward(tmp)              # HyperbolicDecoder
            tmp = m[2](tmp, edges[i])            # GIN
            output.append(tmp)

        x = torch.cat(output, dim=1)   # [num_nodes, output_dim]
        return x

class HIPPI(nn.Module):
    def __init__(
        self,
        input_dim,
        args=None,
        act='relu',
        layer_num=2,
        radius=None,
        dropout=0.0,
        if_bias=True,
        use_att=0,
        local_agg=0,
        feature_fusion='CnM',
        class_num=7,
        in_len=512,
        device=None,
    ):
        super(HIPPI, self).__init__()
        self.feature_fusion = feature_fusion
        self.class_num = class_num

        self.encoder = HIPPIEncoder(
            input_dim=input_dim,
            args=args,
            act=act,
            layer_num=layer_num,
            radius=radius,
            dropout=dropout,
            if_bias=if_bias,
            use_att=use_att,
            local_agg=local_agg,
            class_num=class_num,
            in_len=in_len,
            device=device,
        )

        hidden3 = self.encoder.output_dim
        self.GatedNetwork = GatedInteractionNetwork(hidden3, hidden3, hidden3)

        fc2_dim = hidden3 * 1
        self.fc2 = nn.Sequential(
            nn.Linear(fc2_dim, int(fc2_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(fc2_dim / 2), int(fc2_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(fc2_dim / 4), class_num),
        )

    def forward(self, data, edge_id=None):
        x_all = self.encoder(data)         # [num_nodes, hidden_dim]
        edge_index = data.edge2
        node_id = edge_index[:, edge_id]

        if node_id.dim() == 1:
            x1 = x_all[node_id[0]].unsqueeze(0)
            x2 = x_all[node_id[1]].unsqueeze(0)
        else:
            x1 = x_all[node_id[0]]
            x2 = x_all[node_id[1]]

        x = self.GatedNetwork(x1, x2)
        x = self.fc2(x)
        return x

class HyperbolicDecoder(nn.Module):
    """
    Decoder for node-level representation: hyperbolic -> tangent -> linear.
    """

    def __init__(self, input_dim, output_dim, if_bias, dropout, radius):
        super(HyperbolicDecoder, self).__init__()
        self.manifold = src.mainfold.Hyperboloid()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.if_bias = if_bias
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=self.if_bias)
        self.radius = radius

    def forward(self, x):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.radius), c=self.radius)
        x = self.linear(x)
        return x

class GatedInteractionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatedInteractionNetwork, self).__init__()
        self.fc_interaction = nn.Linear(input_dim, hidden_dim)
        self.fc_gate = nn.Linear(input_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1, x2):
        interaction = x1 * x2
        gate = torch.sigmoid(self.fc_gate(x1 + x2))
        gated_interaction = gate * F.relu(self.fc_interaction(interaction))
        output = self.fc_output(gated_interaction)
        return output

class FactorizedBilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, factor_dim=256):
        super(FactorizedBilinearPooling, self).__init__()
        self.W1 = nn.Linear(input_dim1, factor_dim, bias=False)
        self.W2 = nn.Linear(input_dim2, factor_dim, bias=False)
        self.fc = nn.Linear(factor_dim, output_dim)

    def forward(self, v1, v2):
        v1_transformed = self.W1(v1)
        v2_transformed = self.W2(v2)
        factorized_interaction = v1_transformed * v2_transformed
        output = self.fc(factorized_interaction)
        return output

class GatedBilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(GatedBilinearPooling, self).__init__()
        self.bilinear_layer = nn.Bilinear(input_dim1, input_dim2, output_dim)
        self.gate_layer1 = nn.Linear(input_dim1, output_dim)
        self.gate_layer2 = nn.Linear(input_dim2, output_dim)

    def forward(self, v1, v2):
        bilinear_output = self.bilinear_layer(v1, v2)
        gate_v1 = self.gate_layer1(v1)
        gate_v2 = self.gate_layer2(v2)
        gate = torch.sigmoid(gate_v1 + gate_v2)
        gated_bilinear_output = bilinear_output * gate
        return gated_bilinear_output

# ======================
# ======================

class CodeBook(nn.Module):
    def __init__(self, param, data_loader):
        super(CodeBook, self).__init__()
        self.param = param
        self.Protein_Encoder = GCN_Encoder(param, data_loader)
        self.Protein_Decoder = GCN_Decoder(param)
        self.vq_layer = VectorQuantizer(
            param['prot_hidden_dim'],
            param['num_embeddings'],
            param['commitment_cost'],
        )

    def forward(self, batch_graph):
        z = self.Protein_Encoder.encoding(batch_graph)
        e, e_q_loss, encoding_indices = self.vq_layer(z)

        x_recon = self.Protein_Decoder.decoding(batch_graph, e)
        recon_loss = F.mse_loss(x_recon, batch_graph.ndata['x'])

        device = z.device
        mask = torch.bernoulli(
            torch.full(size=(self.param['num_embeddings'],), fill_value=self.param['mask_ratio'])
        ).bool().to(device)
        mask_index = mask[encoding_indices]
        e[mask_index] = 0.0

        x_mask_recon = self.Protein_Decoder.decoding(batch_graph, e)

        x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
        y = F.normalize(batch_graph.ndata['x'][mask_index], p=2, dim=-1, eps=1e-12)
        mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(self.param['sce_scale']))

        return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)

def get_classifier(hidden_layer, class_num, feature_fusion):
    fc = None
    if feature_fusion == 'CnM':
        fc = nn.Linear(3 * hidden_layer, class_num)
    elif feature_fusion == 'concat':
        fc = nn.Linear(2 * hidden_layer, class_num)
    elif feature_fusion == 'mul':
        fc = nn.Linear(1 * hidden_layer, class_num)
    return fc

class HyperbolicGCN(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        c_in,
        c_out,
        dropout,
        act,
        use_bias,
        use_att,
        local_agg,
    ):
        super(HyperbolicGCN, self).__init__()
        self.manifold = manifold
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, c={self.c}'

class HypAgg(nn.Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return f'c={self.c}'

class HypAct(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return f'c_in={self.c_in}, c_out={self.c_out}'

def get_mainfold():
    return src.mainfold.Hyperboloid()

class GCN_Encoder(nn.Module):
    def __init__(self, param, data_loader):
        super(GCN_Encoder, self).__init__()
        self.data_loader = data_loader
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
        self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
        self.layers.append(
            HeteroGraphConv(
                {
                    'SEQ': GraphConv(param['input_dim'], param['prot_hidden_dim']),
                    'STR_KNN': GraphConv(param['input_dim'], param['prot_hidden_dim']),
                    'STR_DIS': GraphConv(param['input_dim'], param['prot_hidden_dim']),
                },
                aggregate='sum',
            )
        )

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
            self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
            self.layers.append(
                HeteroGraphConv(
                    {
                        'SEQ': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                        'STR_KNN': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                        'STR_DIS': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                    },
                    aggregate='sum',
                )
            )

    def forward(self, vq_layer):
        prot_embed_list = []
        for iter, batch_graph in enumerate(self.data_loader):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_graph = batch_graph.to(device)
            h = self.encoding(batch_graph)
            z, _, _ = vq_layer(h)
            batch_graph.ndata['h'] = torch.cat([h, z], dim=-1)
            prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
            prot_embed_list.append(prot_embed)

        return torch.cat(prot_embed_list, dim=0)

    def encoding(self, batch_graph):
        x = batch_graph.ndata['x']
        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.norms[l](F.relu(self.fc[l](x['amino_acid'])))
            if l != self.num_layers - 1:
                x = self.dropout(x)
        return x

class GCN_Decoder(nn.Module):
    def __init__(self, param):
        super(GCN_Decoder, self).__init__()
        self.num_layers = param['prot_num_layers']
        self.dropout = nn.Dropout(param['dropout_ratio'])
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
            self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
            self.layers.append(
                HeteroGraphConv(
                    {
                        'SEQ': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                        'STR_KNN': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                        'STR_DIS': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                    },
                    aggregate='sum',
                )
            )

        self.fc.append(nn.Linear(param['prot_hidden_dim'], param['input_dim']))
        self.layers.append(
            HeteroGraphConv(
                {
                    'SEQ': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                    'STR_KNN': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                    'STR_DIS': GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']),
                },
                aggregate='sum',
            )
        )

    def decoding(self, batch_graph, x):
        for l, layer in enumerate(self.layers):
            x = layer(batch_graph, {'amino_acid': x})
            x = self.fc[l](x['amino_acid'])
            if l != self.num_layers - 1:
                x = self.dropout(self.norms[l](F.relu(x)))
        return x

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer.
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        encoding_indices = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices)

        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss, encoding_indices

    def get_code_indices(self, x):
        emb_w = F.normalize(self.embeddings.weight, p=2, dim=-1)
        distances = (
            torch.sum(x ** 2, dim=-1, keepdim=True)
            + torch.sum(emb_w ** 2, dim=1)
            - 2.0 * torch.matmul(x, emb_w.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

    def quantize(self, encoding_indices):
        return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)

# ================
# ================

class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.fc = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_tangent, adj):
        # x_tangent: [N, F]
        # adj: sparse matrix [N, N]
        scores = self.fc(x_tangent)  # [N, 1]
        scores = scores.squeeze(-1)
        scores = torch.sigmoid(scores)
        dense_adj = adj.to_dense()
        att_adj = dense_adj * scores.unsqueeze(0)
        row_sum = att_adj.sum(-1, keepdim=True) + 1e-12
        att_adj = att_adj / row_sum
        return att_adj

# =====================================
# =====================================

class HIPPI_VAE_Latent(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, hidden_dim)
        self.fc_logvar = nn.Linear(in_dim, hidden_dim)

        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, h):
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        logvar = torch.clamp(logvar, min=-10, max=2)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        )
        kl_loss = kl_per_sample.mean()

        return z, kl_loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # [max_len, 1, d_model]

    def forward(self, x):
        """
        x: [seq_len, batch, d_model]
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, ff_dim, num_head):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_head, batch_first=False)
        self.linear1 = nn.Linear(dim, ff_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask=None):
        # src: [S, N, E]
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dim, ff_dim, num_head) for _ in range(num_layer)]
        )

    def forward(self, src, src_key_padding_mask=None):
        """
        src: [S, N, E]
        src_key_padding_mask: [N, S]
        """
        out = src
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask)
        return out

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, ff_dim, num_head):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_head, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(dim, num_head, batch_first=False)
        self.linear1 = nn.Linear(dim, ff_dim)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_key_padding_mask=None,
        incremental_state=None,
    ):
        # tgt: [T, N, E]
        # memory: [S, N, E]
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Decoder(nn.Module):

    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(dim, ff_dim, num_head) for _ in range(num_layer)]
        )

    def forward(self, tgt, memory, x_mask=None, mem_padding_mask=None):
        # tgt: [T, B, H]
        # memory: [S, B, H]
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=x_mask, memory_key_padding_mask=mem_padding_mask)
        return out

    def forward_one(self, tgt, memory, incremental_state, mem_padding_mask=None):
        raise NotImplementedError("Use full sequence decoding in _generate instead")

# ==========================
# ==========================

class PPIGEN(BaseModel):

    def __init__(
        self,
        input_dim,
        vocab_size,
        pad_value,
        sos_value=None,
        eos_value=None,
        act='relu',
        layer_num=2,
        radius=None,
        dropout=0.0,
        if_bias=True,
        use_att=0,
        local_agg=0,
        class_num=7,
        in_len=512,
        device=None,
        hidden_dim=376,
        ff_dim=1024,
        heads=8,
        layers=4,
        max_len=4096,
        **args
    ):
        super(PPIGEN, self).__init__(**args)
        self.device = device
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.class_num = class_num

        self.encoder = HIPPIEncoder(
            input_dim=input_dim,
            args=self.args,
            act=act,
            layer_num=layer_num,
            radius=radius,
            dropout=dropout,
            if_bias=if_bias,
            use_att=use_att,
            local_agg=local_agg,
            class_num=class_num,
            in_len=in_len,
            device=device,
        )
        self.node_dim = self.encoder.output_dim

        self.GatedNetwork = GatedInteractionNetwork(self.node_dim, self.node_dim, self.node_dim)
        fc2_dim = self.node_dim * 1
        self.fc2 = nn.Sequential(
            nn.Linear(fc2_dim, int(fc2_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(fc2_dim / 2), int(fc2_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(fc2_dim / 4), class_num),
        )

        self.latent = HIPPI_VAE_Latent(in_dim=self.node_dim, hidden_dim=hidden_dim)

        self.dencoder = TransformerEncoder(dim=hidden_dim, ff_dim=ff_dim, num_head=heads, num_layer=layers)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_len)
        self.zz_seg_encoding = nn.Parameter(torch.randn(hidden_dim))

        self.decoder = Decoder(dim=hidden_dim, ff_dim=ff_dim, num_head=heads, num_layer=layers)

        self.vocab_size = vocab_size
        self.word_embed = nn.Embedding(vocab_size, hidden_dim)
        self.word_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )
        torch.nn.init.zeros_(self.word_pred[3].bias)

        self.pad_value = pad_value
        self.sos_value = sos_value if sos_value is not None else 0
        self.eos_value = eos_value if eos_value is not None else pad_value
        self.kl_anneal_start = 0.0
        self.kl_anneal_end = 1.0
        self.kl_warmup_steps = 5000
        self.max_kl_weight = 0.01
        self.register_buffer('_step_counter', torch.tensor(0))

    def expand_then_fusing(self, z):
        """
        z: [batch, hidden_dim]
        """
        zz = z + self.zz_seg_encoding  # [batch, hidden_dim]
        zz = zz.unsqueeze(0)           # [1, batch, hidden_dim]
        full_mask = zz.new_zeros(zz.shape[1], zz.shape[0]).bool()
        zzz = self.dencoder(zz, src_key_padding_mask=full_mask)  # [1, batch, hidden_dim]
        return zzz, full_mask

    def get_kl_weight(self):
        if not self.training:
            return self.max_kl_weight

        step = self._step_counter.item()
        if step < self.kl_warmup_steps:
            return self.max_kl_weight * (step / self.kl_warmup_steps)
        else:
            return self.max_kl_weight

    def forward(self, data, edge_id, target_seq=None):
        """
        - embed1 / sparse_adj1 / edge1 / edge2
        """

        x_all = self.encoder(data)
        edge_index = data.edge2
        node_id = edge_index[:, edge_id]  # [2] or [2, B]

        if node_id.dim() == 1:
            x1 = x_all[node_id[0]].unsqueeze(0)
            x2 = x_all[node_id[1]].unsqueeze(0)
        else:
            x1 = x_all[node_id[0]]
            x2 = x_all[node_id[1]]

        inter = self.GatedNetwork(x1, x2)  # [batch, node_dim]
        pair_logits = self.fc2(inter)      # [batch, class_num]

        z, kl_loss_raw = self.latent(inter)    # [batch, hidden_dim]

        kl_weight = self.get_kl_weight()
        kl_loss = kl_loss_raw * kl_weight

        if self.training:
            self._step_counter += 1

        zzz, encoder_mask = self.expand_then_fusing(z)

        if target_seq is not None:
            targets = target_seq           # [batch, T]
            B, T = targets.shape

            # causal mask: [T, T]
            target_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=targets.device), diagonal=1
            )

            target_embed = self.word_embed(targets)  # [B, T, H]
            target_embed = target_embed.permute(1, 0, 2).contiguous()  # [T, B, H]
            target_embed = self.pos_encoding(target_embed)

            out = self.decoder(
                target_embed,
                zzz,
                x_mask=target_mask,
                mem_padding_mask=encoder_mask,
            )
            out = out.permute(1, 0, 2).contiguous()  # [B, T, H]
            prediction_scores = self.word_pred(out)  # [B, T, vocab]

            shifted_scores = prediction_scores[:, :-1, :].contiguous()
            shifted_targets = targets[:, 1:].contiguous()
            B2, T2, V = shifted_scores.size()
            shifted_scores = shifted_scores.view(-1, V)
            shifted_targets = shifted_targets.view(-1)

            lm_loss = F.cross_entropy(
                shifted_scores,
                shifted_targets,
                ignore_index=self.pad_value,
            )
        else:
            prediction_scores = None
            lm_loss = torch.tensor(0.0, device=z.device)

        return pair_logits, prediction_scores, lm_loss, kl_loss

    def _generate(self, zzz, encoder_mask, random_sample=False, return_score=False):
        batch_size = zzz.shape[1]
        device = zzz.device

        generated = torch.full((batch_size, 1), self.sos_value, dtype=torch.long, device=device)

        if return_score:
            all_scores = []

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(self.max_len - 1):
            current_len = generated.size(1)

            token_embed = self.word_embed(generated)  # [B, current_len, H]
            pos_embed = self.pos_encoding.pe[:current_len].transpose(0, 1)  # [1, current_len, H] -> [current_len, 1, H]
            target_embed = token_embed + pos_embed.squeeze(1)  # [B, current_len, H]
            target_embed = target_embed.permute(1, 0, 2)  # [current_len, B, H]

            causal_mask = torch.triu(
                torch.ones(current_len, current_len, dtype=torch.bool, device=device),
                diagonal=1
            )

            decoder_out = self.decoder(
                target_embed,
                zzz,
                x_mask=causal_mask,
                mem_padding_mask=encoder_mask
            )  # [current_len, B, H]

            last_hidden = decoder_out[-1]  # [B, H]
            logits = self.word_pred(last_hidden)  # [B, vocab_size]

            if return_score:
                all_scores.append(logits)

            if random_sample:
                temperature = 0.9
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(1)  # [B]
            else:
                next_token = torch.argmax(logits, dim=-1)  # [B]

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)  # [B, current_len+1]

            finished |= (next_token == self.eos_value) | (next_token == self.pad_value)
            if finished.all():
                break

        predict = generated[:, 1:]

        if return_score:
            return predict, torch.stack(all_scores, dim=1)
        return predict

    def generate(self, data, edge_id, random_sample=False, return_z=False):
        x_all = self.encoder(data)
        edge_index = data.edge2
        node_id = edge_index[:, edge_id]

        if node_id.dim() == 1:
            x1 = x_all[node_id[0]].unsqueeze(0)
            x2 = x_all[node_id[1]].unsqueeze(0)
        else:
            x1 = x_all[node_id[0]]
            x2 = x_all[node_id[1]]

        inter = self.GatedNetwork(x1, x2)
        z, kl_loss = self.latent(inter)
        zzz, encoder_mask = self.expand_then_fusing(z)

        predict = self._generate(zzz, encoder_mask, random_sample=random_sample, return_score=False)
        if return_z:
            return predict, z.detach().cpu().numpy()
        return predict