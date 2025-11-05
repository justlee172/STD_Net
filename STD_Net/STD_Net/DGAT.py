import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import time
import multiprocessing
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import copy

class conGraphConvolutionlayer(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(conGraphConvolutionlayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # Initialize the weights and biases using a uniform distribution
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # Defined the string representation of the model, which is used for printing and debugging.

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import time
import multiprocessing
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import copy
from torch_geometric.nn import GATv2Conv


class ImprovedGraphConv(Module):

    def __init__(self, in_features, out_features, heads=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.conv = GATv2Conv(in_features, out_features, heads=heads, concat=False)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features}->{self.out_features}, heads={self.heads})'

# Convert the dense adjacency matrix to the edge index format
def dense_to_edge_index(adj):


    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.float)


    indices = adj.nonzero(as_tuple=False).t()

    if indices.nelement() == 0:
        return torch.empty((2, 0), dtype=torch.long)


    max_index = adj.shape[0] - 1
    valid_mask = (indices[0] <= max_index) & (indices[1] <= max_index)
    valid_indices = indices[:, valid_mask]

    return valid_indices.contiguous()


class conDGAT(nn.Module):
    def __init__(self, nfeat,
                 nhid,
                 common_hid_layers_num,
                 fcnn_hid_layers_num,
                 dropout,
                 nout1,
                 use_gatv2=False,
                 gat_heads=4
                 ):
        super(conDGAT, self).__init__()
        self.input_norm = nn.LayerNorm(nfeat)
        self.nfeat = nfeat
        self.nhid = nhid
        self.common_hid_layers_num = common_hid_layers_num
        self.fcnn_hid_layers_num = fcnn_hid_layers_num
        self.nout1 = nout1
        self.dropout = dropout
        self.training = True
        self.use_gatv2 = use_gatv2
        self.gat_heads = gat_heads

        if use_gatv2:
            GraphConvLayer = ImprovedGraphConv
        else:
            GraphConvLayer = conGraphConvolutionlayer

        ## The beginning layer
        self.gc_in_exp = GraphConvLayer(nfeat, nhid, heads=gat_heads)
        self.bn_node_in_exp = nn.BatchNorm1d(nhid)

        self.gc_in_sp = GraphConvLayer(nfeat, nhid, heads=gat_heads)
        self.bn_node_in_sp = nn.BatchNorm1d(nhid)

        ## common_hid_layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec('self.cgc{}_exp = GraphConvLayer(nhid, nhid, heads=gat_heads)'.format(i + 1))
                exec('self.bn_node_chid{}_exp = nn.BatchNorm1d(nhid)'.format(i + 1))
                exec('self.cgc{}_sp = GraphConvLayer(nhid, nhid, heads=gat_heads)'.format(i + 1))
                exec('self.bn_node_chid{}_sp = nn.BatchNorm1d(nhid)'.format(i + 1))

        ## FCNN layers
        self.gc_out11 = nn.Linear(2 * nhid, nhid, bias=True)
        self.bn_out1 = nn.BatchNorm1d(nhid)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec('self.gc_out11{} = nn.Linear(nhid, nhid, bias=True)'.format(i + 1))
                exec('self.bn_out11{} = nn.BatchNorm1d(nhid)'.format(i + 1))

        self.gc_out12 = nn.Linear(nhid, nout1, bias=True)

    def forward(self, x, adjs):
        self.x = self.input_norm(x)

        # Convert the adjacency matrix to the edge index format (if it is GATv2)
        if self.use_gatv2:
            edge_index_exp = dense_to_edge_index(adjs[0])
            edge_index_sp = dense_to_edge_index(adjs[1])
        else:
            edge_index_exp = adjs[0]
            edge_index_sp = adjs[1]

        ## input layer
        # ex
        self.x_exp = self.gc_in_exp(self.x, edge_index_exp)
        self.x_exp = self.bn_node_in_exp(self.x_exp)
        self.x_exp = F.elu(self.x_exp)
        self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
        # sp
        self.x_sp = self.gc_in_sp(self.x, edge_index_sp)
        self.x_sp = self.bn_node_in_sp(self.x_sp)
        self.x_sp = F.elu(self.x_sp)
        self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        ## common layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                # Residual
                residual_exp = self.x_exp
                residual_sp = self.x_sp

                exec("self.x_exp = self.cgc{}_exp(self.x_exp, edge_index_exp)".format(i + 1))
                exec("self.x_exp = self.bn_node_chid{}_exp(self.x_exp)".format(i + 1))
                self.x_exp = F.elu(self.x_exp + residual_exp)  # residual connection
                self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)

                exec("self.x_sp = self.cgc{}_sp(self.x_sp, edge_index_sp)".format(i + 1))
                exec("self.x_sp = self.bn_node_chid{}_sp(self.x_sp)".format(i + 1))
                self.x_sp = F.elu(self.x_sp + residual_sp)  # residual connection
                self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        ## FCNN layers
        self.x1 = torch.cat([self.x_exp, self.x_sp], dim=1)

        self.x1 = self.gc_out11(self.x1)
        self.x1 = self.bn_out1(self.x1)
        self.x1 = F.elu(self.x1)
        self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec("self.x1 = self.gc_out11{}(self.x1)".format(i + 1))
                exec("self.x1 = self.bn_out11{}(self.x1)".format(i + 1))
                self.x1 = F.elu(self.x1)
                self.x1 = F.dropout(self.x1, self.dropout, training=self.training)

        self.x1 = self.gc_out12(self.x1)

        gc_list = {}
        gc_list['gc_in_exp'] = self.gc_in_exp
        gc_list['gc_in_sp'] = self.gc_in_sp
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("gc_list['cgc{}_exp'] = self.cgc{}_exp".format(i + 1, i + 1))
                exec("gc_list['cgc{}_sp'] = self.cgc{}_sp".format(i + 1, i + 1))
        gc_list['gc_out11'] = self.gc_out11
        if self.fcnn_hid_layers_num > 0:
            exec("gc_list['gc_out11{}'] =  self.gc_out11{}".format(i + 1, i + 1))
        gc_list['gc_out12'] = self.gc_out12

        self.layer_outputs = {
            'gc_in_exp': self.x_exp,
            'gc_in_sp': self.x_sp,
            'final_embedding': self.x1
        }

        return F.log_softmax(self.x1, dim=1), gc_list


def conDGAT_train(model,
                 train_valid_len,
                 test_len,
                 feature,
                 adjs,
                 label,
                 epoch_n,
                 loss_fn,
                 optimizer,
                 train_valid_ratio=0.9,
                 scheduler=None,
                 early_stopping_patience=5,
                 clip_grad_max_norm=1,
                 load_test_groundtruth=False,
                 print_epoch_step=1,
                 cpu_num=-1,
                 DGAT_device='GPU'
                 ):

    use_cuda = (DGAT_device != 'CPU') and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device: {device}')

    if torch.__version__<"2.0.0":
        def round_decimals(tensor,decimals):
            scale=10**decimals
            return (tensor*scale).round().div(scale)
    else:
        def round_decimals(tensor,decimals):
            return torch.round(tensor,decimals=decimals)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)

    model = model.to(device)

    if model.use_gatv2:
        adjs = [dense_to_edge_index(adj).to(device) for adj in adjs]
    else:
        adjs = [adj.to(device) for adj in adjs]

    feature = feature.to(device)
    label = label.to(device)

    time_open = time.time()

    train_idx = range(int(train_valid_len * train_valid_ratio))
    valid_idx = range(len(train_idx), train_valid_len)
    best_val = np.inf
    clip = 0
    loss = []
    para_list = []
    for epoch in range(epoch_n):

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        optimizer.zero_grad()

        # === Use automatic mixed precision ===
        with torch.cuda.amp.autocast(enabled=use_cuda):
            output1, paras = model(feature.float(), adjs)

            loss_train1 = loss_fn(output1[list(np.array(train_idx) + test_len)],
                                  label[list(np.array(train_idx) + test_len)].float())
            loss_val1 = loss_fn(output1[list(np.array(valid_idx) + test_len)],
                                label[list(np.array(valid_idx) + test_len)].float())
            if load_test_groundtruth == True:
                loss_test1 = loss_fn(output1[:test_len], label[:test_len].float())
                loss.append([loss_train1.item(), loss_val1.item(), loss_test1.item()])
            else:
                loss.append([loss_train1.item(), loss_val1.item(), None])

        if epoch % print_epoch_step == 0:
            print("**")
            print("Epoch {}/{}".format(epoch + 1, epoch_n),
                  'loss_train: {:.4f}'.format(loss_train1.item()),
                  'loss_val: {:.4f}'.format(loss_val1.item()),
                  end='\t'
                  )
            if load_test_groundtruth == True:
                print("Test loss= {:.4f}".format(loss_test1.item()), end='\t')
            print('time: {:.4f}s'.format(time.time() - time_open))

        para_list.append(paras.copy())
        for i in paras.keys():
            para_list[-1][i] = copy.deepcopy(para_list[-1][i])

        if early_stopping_patience > 0:

            if round_decimals(loss_val1, decimals=4) < best_val:
                best_val = round_decimals(loss_val1, decimals=4)
                best_paras = paras.copy()
                best_loss = loss.copy()
                clip = 1
                for i in paras.keys():
                    best_paras[i] = copy.deepcopy(best_paras[i])
            else:
                clip += 1
                if clip == early_stopping_patience:
                    break
        else:
            best_loss = loss.copy()
            best_paras = None


        if use_cuda:
            scaler.scale(loss_train1).backward()
            scaler.unscale_(optimizer)


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            scaler.step(optimizer)
            scaler.update()
        else:

            loss_train1.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        if scheduler != None:
            try:
                scheduler.step()
            except:
                scheduler.step(metrics=loss_val1)

    print("*Final Loss*")
    print("Epoch {}/{}".format(epoch + 1, epoch_n),
          'loss_train: {:.4f}'.format(loss_train1.item()),
          'loss_val: {:.4f}'.format(loss_val1.item()),
          end='\t'
          )
    if load_test_groundtruth == True:
        print("Test loss= {:.4f}".format(loss_test1.item()), end='\t')
    print('time: {:.4f}s'.format(time.time() - time_open))

    torch.cuda.empty_cache()

    return output1.cpu(), loss, model.cpu()


