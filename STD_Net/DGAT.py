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
from .optimization import (
    SparseCompositionalLoss,
    conditional_prototype_loss,
    negative_binomial_nll,
    reconstruct_counts,
    sparsemax,
    support_calibrated_sparsemax,
    spatial_total_variation,
)


class ImprovedGraphConv(Module):

    def __init__(self, in_features, out_features, heads=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.conv = GATv2Conv(
            in_features,
            out_features,
            heads=heads,
            concat=False,
            edge_dim=1,
            add_self_loops=False,
        )

    def forward(self, x, graph):
        edge_index, edge_weight = graph
        edge_attributes = edge_weight.reshape(-1, 1).to(dtype=x.dtype)
        return self.conv(x, edge_index, edge_attr=edge_attributes)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features}->{self.out_features}, heads={self.heads})'


class _GradientReversalFunction(torch.autograd.Function):
    """Identity in the forward pass and sign reversal in the backward pass."""

    @staticmethod
    def forward(ctx, features, coefficient):
        ctx.coefficient = float(coefficient)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class GradientReversal(nn.Module):
    """Small domain-adaptation layer used only by opt-in profiles."""

    def __init__(self, coefficient=1.0):
        super().__init__()
        self.coefficient = float(coefficient)

    def forward(self, features):
        return _GradientReversalFunction.apply(features, self.coefficient)

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


def dense_to_weighted_graph(adj):
    """Convert a dense adjacency matrix without discarding edge weights."""
    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.float)
    indices = adj.nonzero(as_tuple=False).t().contiguous()
    if indices.nelement() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=adj.device),
            torch.empty((0,), dtype=adj.dtype, device=adj.device),
        )
    weights = adj[indices[0], indices[1]].contiguous()
    return indices, weights


class conDGAT(nn.Module):
    def __init__(self, nfeat,
                 nhid,
                 common_hid_layers_num,
                 fcnn_hid_layers_num,
                 dropout,
                 nout1,
                 use_gatv2=False,
                 gat_heads=4,
                 output_transform='softmax',
                 gated_fusion=False,
                 reconstruction_gene_count=0,
                 support_smoothing_epsilon=0.0,
                 domain_adversarial=False,
                 domain_adversarial_lambda=1.0,
                 adaptive_signature=False,
                 adaptive_signature_rank=8,
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
        self.output_transform = output_transform
        self.gated_fusion = gated_fusion
        self.reconstruction_gene_count = int(reconstruction_gene_count)
        self.support_smoothing_epsilon = float(support_smoothing_epsilon)
        self.domain_adversarial = bool(domain_adversarial)
        self.domain_adversarial_lambda = float(domain_adversarial_lambda)
        self.adaptive_signature = bool(adaptive_signature)
        self.adaptive_signature_rank = int(adaptive_signature_rank)

        if output_transform not in {'softmax', 'sparsemax'}:
            raise ValueError("output_transform must be 'softmax' or 'sparsemax'")
        if not 0.0 <= self.support_smoothing_epsilon <= 1.0:
            raise ValueError("support_smoothing_epsilon must lie in [0, 1]")
        if self.domain_adversarial_lambda < 0:
            raise ValueError("domain_adversarial_lambda must be non-negative")
        if self.adaptive_signature_rank < 1:
            raise ValueError("adaptive_signature_rank must be positive")

        if use_gatv2:
            GraphConvLayer = ImprovedGraphConv
        else:
            GraphConvLayer = conGraphConvolutionlayer

        ## The beginning layer
        def graph_layer(in_features, out_features):
            if use_gatv2:
                return GraphConvLayer(in_features, out_features, heads=gat_heads)
            return GraphConvLayer(in_features, out_features)

        self.gc_in_exp = graph_layer(nfeat, nhid)
        self.bn_node_in_exp = nn.BatchNorm1d(nhid)

        self.gc_in_sp = graph_layer(nfeat, nhid)
        self.bn_node_in_sp = nn.BatchNorm1d(nhid)

        ## common_hid_layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                setattr(self, f'cgc{i + 1}_exp', graph_layer(nhid, nhid))
                setattr(self, f'bn_node_chid{i + 1}_exp', nn.BatchNorm1d(nhid))
                setattr(self, f'cgc{i + 1}_sp', graph_layer(nhid, nhid))
                setattr(self, f'bn_node_chid{i + 1}_sp', nn.BatchNorm1d(nhid))

        ## FCNN layers
        if gated_fusion:
            self.fusion_gate = nn.Linear(2 * nhid, nhid, bias=True)
            fusion_width = nhid
        else:
            self.fusion_gate = None
            fusion_width = 2 * nhid
        self.gc_out11 = nn.Linear(fusion_width, nhid, bias=True)
        self.bn_out1 = nn.BatchNorm1d(nhid)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                setattr(self, f'gc_out11{i + 1}', nn.Linear(nhid, nhid, bias=True))
                setattr(self, f'bn_out11{i + 1}', nn.BatchNorm1d(nhid))

        self.gc_out12 = nn.Linear(nhid, nout1, bias=True)
        if self.reconstruction_gene_count > 0:
            self.log_gene_scale = nn.Parameter(
                torch.zeros(self.reconstruction_gene_count)
            )
            self.log_dispersion = nn.Parameter(
                torch.zeros(self.reconstruction_gene_count)
            )
            self.ambient_logit = nn.Parameter(torch.tensor(-4.0))
        else:
            self.register_parameter('log_gene_scale', None)
            self.register_parameter('log_dispersion', None)
            self.register_parameter('ambient_logit', None)

        if self.domain_adversarial:
            self.domain_gradient_reversal = GradientReversal(
                self.domain_adversarial_lambda
            )
            self.domain_classifier = nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.LayerNorm(nhid),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(nhid, 1),
            )
        else:
            self.domain_gradient_reversal = None
            self.domain_classifier = None

        if self.adaptive_signature:
            if self.reconstruction_gene_count <= 0:
                raise ValueError(
                    "adaptive_signature requires reconstruction_gene_count"
                )
            self.signature_context = nn.Linear(
                nhid, self.adaptive_signature_rank
            )
            self.signature_basis = nn.Parameter(
                torch.empty(
                    self.adaptive_signature_rank,
                    nout1,
                    self.reconstruction_gene_count,
                )
            )
            nn.init.normal_(self.signature_basis, mean=0.0, std=0.01)
        else:
            self.signature_context = None
            self.register_parameter('signature_basis', None)

    def reconstruction_parameters(self):
        if self.reconstruction_gene_count <= 0:
            return None, None, None
        gene_scale = F.softplus(self.log_gene_scale) + 1e-4
        dispersion = F.softplus(self.log_dispersion) + 1e-4
        ambient_fraction = torch.sigmoid(self.ambient_logit) * 0.25
        return gene_scale, dispersion, ambient_fraction

    def domain_logits(self, embedding=None):
        """Return real-versus-pseudo logits through a gradient-reversal layer."""
        if self.domain_classifier is None:
            raise RuntimeError(
                "domain adversarial head is disabled for this model"
            )
        if embedding is None:
            embedding = self.final_hidden
        return self.domain_classifier(
            self.domain_gradient_reversal(embedding)
        ).squeeze(-1)

    def adaptive_signatures(self, embedding, reference_signature):
        """Return niche-conditioned signatures in a normalized gene space."""
        if self.signature_context is None or self.signature_basis is None:
            raise RuntimeError("adaptive signature decoder is disabled")
        context = torch.tanh(self.signature_context(embedding))
        delta = torch.einsum(
            "nr,rcg->ncg", context, self.signature_basis
        )
        base = reference_signature.clamp_min(1e-8).log().unsqueeze(0)
        signatures = F.softmax(base + 0.1 * delta, dim=-1)
        return signatures, delta

    def forward(self, x, adjs, hard_output=False):
        self.x = self.input_norm(x)

        graph_exp = adjs[0]
        graph_sp = adjs[1]

        ## input layer
        # ex
        self.x_exp = self.gc_in_exp(self.x, graph_exp)
        self.x_exp = self.bn_node_in_exp(self.x_exp)
        self.x_exp = F.elu(self.x_exp)
        self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
        # sp
        self.x_sp = self.gc_in_sp(self.x, graph_sp)
        self.x_sp = self.bn_node_in_sp(self.x_sp)
        self.x_sp = F.elu(self.x_sp)
        self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        ## common layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                # Residual
                residual_exp = self.x_exp
                residual_sp = self.x_sp

                self.x_exp = getattr(self, f'cgc{i + 1}_exp')(self.x_exp, graph_exp)
                self.x_exp = getattr(self, f'bn_node_chid{i + 1}_exp')(self.x_exp)
                self.x_exp = F.elu(self.x_exp + residual_exp)  # residual connection
                self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)

                self.x_sp = getattr(self, f'cgc{i + 1}_sp')(self.x_sp, graph_sp)
                self.x_sp = getattr(self, f'bn_node_chid{i + 1}_sp')(self.x_sp)
                self.x_sp = F.elu(self.x_sp + residual_sp)  # residual connection
                self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        ## FCNN layers
        concatenated = torch.cat([self.x_exp, self.x_sp], dim=1)
        if self.fusion_gate is not None:
            self.fusion_weights = torch.sigmoid(self.fusion_gate(concatenated))
            self.x1 = (
                self.fusion_weights * self.x_exp
                + (1 - self.fusion_weights) * self.x_sp
            )
        else:
            self.fusion_weights = None
            self.x1 = concatenated

        self.x1 = self.gc_out11(self.x1)
        self.x1 = self.bn_out1(self.x1)
        self.x1 = F.elu(self.x1)
        self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                self.x1 = getattr(self, f'gc_out11{i + 1}')(self.x1)
                self.x1 = getattr(self, f'bn_out11{i + 1}')(self.x1)
                self.x1 = F.elu(self.x1)
                self.x1 = F.dropout(self.x1, self.dropout, training=self.training)

        self.final_hidden = self.x1
        self.logits = self.gc_out12(self.final_hidden)

        gc_list = {}
        gc_list['gc_in_exp'] = self.gc_in_exp
        gc_list['gc_in_sp'] = self.gc_in_sp
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                gc_list[f'cgc{i + 1}_exp'] = getattr(self, f'cgc{i + 1}_exp')
                gc_list[f'cgc{i + 1}_sp'] = getattr(self, f'cgc{i + 1}_sp')
        gc_list['gc_out11'] = self.gc_out11
        if self.fcnn_hid_layers_num > 0:
            for layer_index in range(self.fcnn_hid_layers_num):
                layer_name = f'gc_out11{layer_index + 1}'
                gc_list[layer_name] = getattr(self, layer_name)
        gc_list['gc_out12'] = self.gc_out12

        self.layer_outputs = {
            'gc_in_exp': self.x_exp,
            'gc_in_sp': self.x_sp,
            'final_embedding': self.final_hidden,
            'logits': self.logits,
        }
        if self.output_transform == 'sparsemax':
            epsilon = 0.0 if hard_output else self.support_smoothing_epsilon
            self.probabilities = support_calibrated_sparsemax(
                self.logits,
                epsilon=epsilon,
                dim=1,
            )
            log_probabilities = self.probabilities.clamp_min(1e-8).log()
        else:
            log_probabilities = F.log_softmax(self.logits, dim=1)
            self.probabilities = log_probabilities.exp()

        return log_probabilities, gc_list


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
                 DGAT_device='GPU',
                 objective_config=None,
                 real_counts=None,
                 reference_signature=None,
                 real_spatial_edges=None,
                 random_seed=0,
                 ):
    use_cuda = (DGAT_device != 'CPU') and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device: {device}')
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    if cpu_num == -1:
        torch.set_num_threads(multiprocessing.cpu_count())
    else:
        torch.set_num_threads(max(1, cpu_num))

    model = model.to(device)

    def prepare_graph(graph):
        if isinstance(graph, tuple) and len(graph) == 2:
            return tuple(component.to(device) for component in graph)
        if model.use_gatv2:
            return tuple(
                component.to(device) for component in dense_to_weighted_graph(graph)
            )
        return graph.to(device)

    adjs = [prepare_graph(graph) for graph in adjs]
    feature = feature.to(device)
    label = label.to(device)
    if real_counts is not None:
        real_counts = torch.as_tensor(real_counts, dtype=torch.float32, device=device)
    if reference_signature is not None:
        reference_signature = torch.as_tensor(
            reference_signature, dtype=torch.float32, device=device
        )
    if real_spatial_edges is not None:
        real_spatial_edges = tuple(
            torch.as_tensor(component, device=device) for component in real_spatial_edges
        )

    config = dict(objective_config or {})
    use_composite_objective = bool(config.get('enabled', False))
    composition_loss = SparseCompositionalLoss(
        **dict(config.get('composition', {}))
    ).to(device)
    reconstruction_weight = float(config.get('reconstruction_weight', 0.0))
    prototype_weight = float(config.get('prototype_weight', 0.0))
    spatial_weight = float(config.get('spatial_weight', 0.0))
    domain_adversarial_weight = float(
        config.get('domain_adversarial_weight', 0.0)
    )
    adaptive_signature_weight = float(
        config.get('adaptive_signature_weight', 0.0)
    )
    confidence_threshold = float(config.get('confidence_threshold', 0.35))
    if domain_adversarial_weight > 0 and model.domain_classifier is None:
        raise ValueError(
            "domain_adversarial_weight requires domain_adversarial=True "
            "when constructing conDGAT"
        )
    domain_labels = None
    if domain_adversarial_weight > 0:
        domain_labels = torch.cat(
            [
                torch.zeros(test_len, dtype=torch.float32, device=device),
                torch.ones(train_valid_len, dtype=torch.float32, device=device),
            ]
        )
        domain_loss_fn = nn.BCEWithLogitsLoss()

    # Deterministic stratification by the dominant pseudo-spot cell type.
    pseudo_targets = label[test_len:test_len + train_valid_len].detach().cpu().numpy()
    dominant_types = np.argmax(pseudo_targets, axis=1)
    rng = np.random.default_rng(random_seed)
    train_local = []
    valid_local = []
    validation_fraction = 1 - train_valid_ratio
    for cell_type in np.unique(dominant_types):
        group = np.flatnonzero(dominant_types == cell_type)
        rng.shuffle(group)
        if len(group) <= 1 or validation_fraction <= 0:
            train_local.extend(group.tolist())
            continue
        validation_count = max(1, int(round(len(group) * validation_fraction)))
        validation_count = min(validation_count, len(group) - 1)
        valid_local.extend(group[:validation_count].tolist())
        train_local.extend(group[validation_count:].tolist())
    if not valid_local:
        shuffled = np.arange(train_valid_len)
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(train_valid_len * validation_fraction)))
        valid_local = shuffled[:validation_count].tolist()
        train_local = shuffled[validation_count:].tolist()
    train_indices = torch.as_tensor(
        np.asarray(train_local, dtype=np.int64) + test_len,
        dtype=torch.long,
        device=device,
    )
    valid_indices = torch.as_tensor(
        np.asarray(valid_local, dtype=np.int64) + test_len,
        dtype=torch.long,
        device=device,
    )

    def supervised_loss(output, indices):
        if use_composite_objective:
            return composition_loss(output[indices], label[indices].float())
        return loss_fn(output[indices], label[indices].float())

    def full_training_loss(output):
        components = {}
        components['supervised'] = supervised_loss(output, train_indices)
        total = components['supervised']
        real_probabilities = model.probabilities[:test_len]

        if (
            reconstruction_weight > 0
            and real_counts is not None
            and reference_signature is not None
        ):
            gene_scale, dispersion, ambient_fraction = (
                model.reconstruction_parameters()
            )
            if gene_scale is None or dispersion is None:
                raise ValueError(
                    "reconstruction objective requires model reconstruction parameters"
                )
            signature_for_reconstruction = reference_signature
            adaptive_delta = None
            if model.signature_context is not None:
                signature_for_reconstruction, adaptive_delta = (
                    model.adaptive_signatures(
                        model.layer_outputs['final_embedding'][:test_len],
                        reference_signature,
                    )
                )
            reconstructed = reconstruct_counts(
                real_probabilities,
                signature_for_reconstruction,
                real_counts.sum(dim=1),
                gene_scale=gene_scale,
                ambient_fraction=ambient_fraction,
            )
            components['reconstruction'] = negative_binomial_nll(
                real_counts, reconstructed, dispersion.unsqueeze(0)
            )
            total = total + reconstruction_weight * components['reconstruction']
            if adaptive_delta is not None and adaptive_signature_weight > 0:
                components['adaptive_signature'] = adaptive_delta.pow(2).mean()
                total = total + adaptive_signature_weight * components[
                    'adaptive_signature'
                ]

        if prototype_weight > 0:
            embeddings = model.layer_outputs['final_embedding']
            components['prototype'] = conditional_prototype_loss(
                embeddings[train_indices],
                label[train_indices].float(),
                embeddings[:test_len],
                real_probabilities,
                confidence_threshold=confidence_threshold,
            )
            total = total + prototype_weight * components['prototype']

        if spatial_weight > 0 and real_spatial_edges is not None:
            components['spatial'] = spatial_total_variation(
                real_probabilities,
                real_spatial_edges[0].long(),
                real_spatial_edges[1].float(),
            )
            total = total + spatial_weight * components['spatial']
        if domain_adversarial_weight > 0:
            domain_logits = model.domain_logits(
                model.layer_outputs['final_embedding']
            )
            components['domain'] = domain_loss_fn(domain_logits, domain_labels)
            total = total + domain_adversarial_weight * components['domain']
        components['total'] = total
        return total, components

    time_open = time.time()
    best_val = np.inf
    best_epoch = -1
    best_state_dict = copy.deepcopy(model.state_dict())
    best_loss = []
    loss = []
    component_history = []
    patience_counter = 0
    min_delta = float(config.get('early_stopping_min_delta', 1e-5))

    for epoch in range(epoch_n):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_cuda):
            output_train, _ = model(feature.float(), adjs)
            loss_train_total, training_components = full_training_loss(output_train)

        if use_cuda:
            scaler.scale(loss_train_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad_max_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_train_total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_grad_max_norm
            )
            optimizer.step()

        # Validation and checkpointing happen after the optimizer update.
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_cuda):
                output_eval, _ = model(feature.float(), adjs)
                loss_train_supervised = supervised_loss(output_eval, train_indices)
                loss_val = supervised_loss(output_eval, valid_indices)
                if load_test_groundtruth:
                    loss_test = supervised_loss(
                        output_eval, torch.arange(test_len, device=device)
                    )
                else:
                    loss_test = None

        train_value = float(loss_train_supervised.detach().cpu())
        validation_value = float(loss_val.detach().cpu())
        test_value = (
            float(loss_test.detach().cpu()) if loss_test is not None else None
        )
        loss.append([train_value, validation_value, test_value])
        component_history.append(
            {
                'epoch': epoch + 1,
                **{
                    name: float(value.detach().cpu())
                    for name, value in training_components.items()
                },
                'validation_supervised': validation_value,
            }
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_value)
            else:
                scheduler.step()

        if epoch % print_epoch_step == 0:
            print("**")
            print(
                "Epoch {}/{}".format(epoch + 1, epoch_n),
                'loss_train: {:.4f}'.format(train_value),
                'loss_val: {:.4f}'.format(validation_value),
                end='\t',
            )
            if test_value is not None:
                print("Test loss= {:.4f}".format(test_value), end='\t')
            print('time: {:.4f}s'.format(time.time() - time_open))

        if validation_value < best_val - min_delta:
            best_val = validation_value
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            best_loss = loss.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if (
                early_stopping_patience > 0
                and patience_counter >= early_stopping_patience
            ):
                break

    if best_epoch < 0:
        best_epoch = len(loss) - 1
        best_state_dict = copy.deepcopy(model.state_dict())
        best_loss = loss.copy()
    model.load_state_dict(best_state_dict)
    model.eval()
    model.training_history_components = component_history
    model.best_epoch = best_epoch + 1
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_cuda):
            output1, _ = model(feature.float(), adjs, hard_output=True)

    best_train_loss, best_validation_loss, best_test_loss = best_loss[-1]
    print("*Best Validation Checkpoint*")
    print(
        "Epoch {}/{}".format(best_epoch + 1, epoch_n),
        'loss_train: {:.4f}'.format(best_train_loss),
        'loss_val: {:.4f}'.format(best_validation_loss),
        end='\t',
    )
    if best_test_loss is not None:
        print("Test loss= {:.4f}".format(best_test_loss), end='\t')
    print('time: {:.4f}s'.format(time.time() - time_open))

    if use_cuda:
        torch.cuda.empty_cache()
    return output1.cpu(), best_loss, model.cpu()
