import math
import torch
import torch.nn as nn


class MLPResidualLayer(nn.Module):
    def __init__(self, dim):
        super(MLPResidualLayer, self).__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return residual + x


class DecoupledQNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions, num_heads):
        super(DecoupledQNetwork, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.resnet = MLPResidualLayer(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_heads = VectorizedLinear(hidden_dim, num_actions, num_heads)
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

    def forward(self, x):
        x = self.input_layer.forward(x)
        x = self.layer_norm(self.resnet.forward(x))
        x = x.unsqueeze(dim=0).repeat(self.num_heads, 1, 1)
        vals = self.output_heads.forward(x).transpose(0, 1)
        return vals


class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class VectorizedLinearHead(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size, num_heads):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.num_heads = num_heads

        self.weight = nn.Parameter(torch.empty(ensemble_size, num_heads, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, num_heads, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, num_heads, batch_size, input_size]
        # weight: [ensemble_size, num_heads, input_size, out_size]
        # out: [ensemble_size, num_heads, batch_size, out_size]
        return x @ self.weight + self.bias


class VectorisedMLPResidualLayer(nn.Module):
    def __init__(self, dim, ensemble_size):
        super(VectorisedMLPResidualLayer, self).__init__()

        self.fc1 = VectorizedLinear(dim, dim, ensemble_size)
        self.fc2 = VectorizedLinear(dim, dim, ensemble_size)

    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return residual + x


class EnsembleDecoupledQNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions, num_heads, ensemble_size):
        super(EnsembleDecoupledQNetwork, self).__init__()
        self.input_layer = VectorizedLinear(state_dim, hidden_dim, ensemble_size)
        self.resnet_layer = VectorisedMLPResidualLayer(hidden_dim, ensemble_size)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_heads = VectorizedLinearHead(hidden_dim, num_actions, ensemble_size, num_heads)
        self.num_heads = num_heads
        self.ensemble_size = ensemble_size

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1).repeat(1, self.ensemble_size, 1)
        x = x.transpose(0, 1)
        x = self.input_layer.forward(x)
        x = self.layer_norm(self.resnet_layer.forward(x))
        x = x.unsqueeze(dim=1).repeat(1, self.num_heads, 1, 1)
        return self.output_heads.forward(x).transpose(1, 2).transpose(0, 1)
