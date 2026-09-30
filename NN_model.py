import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dropout_p=0.0):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        
        self.l1 = nn.Linear(channels, channels)
        self.l2 = nn.Linear(channels, channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        if dropout_p > 0:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.dropout = None
    
    def forward(self, x):
        y = F.softplus(self.l1(x))   
        if self.dropout is not None:
            y = self.dropout(y)
        y = self.l2(y)
        return x + self.alpha * y

class Net(nn.Module):
    def __init__(self, input, output, dropout_p=0.0):
        super(Net, self).__init__()
        
        self.input_layer1 = nn.Linear(input, 16)
        self.hidden_layers0 = nn.Linear(16, 32)
        
        self.hidden_layers1 = ResidualBlock(32, dropout_p=0.0)
        self.hidden_layers2 = ResidualBlock(32, dropout_p=0.0)
        self.hidden_layers3 = ResidualBlock(32, dropout_p=dropout_p)
        self.hidden_layers4 = ResidualBlock(32, dropout_p=dropout_p)
        self.hidden_layers5 = ResidualBlock(32, dropout_p=dropout_p)
        self.hidden_layers6 = ResidualBlock(32, dropout_p=dropout_p)
        self.hidden_layers7 = ResidualBlock(32, dropout_p=0.0)
        
        self.hidden_layers8 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, output)
        
    def forward(self, x):
        o = self.act(self.input_layer1(x))
        o = self.act(self.hidden_layers0(o))
        
        o = self.act(self.hidden_layers1(o))
        o = self.act(self.hidden_layers2(o))
        o = self.act(self.hidden_layers3(o))
        o = self.act(self.hidden_layers4(o))
        o = self.act(self.hidden_layers5(o))
        o = self.act(self.hidden_layers6(o))
        o = self.act(self.hidden_layers7(o))

        
        o = self.act(self.hidden_layers8(o))
            
        opt = self.output_layer(o)
        
        mu_logit = opt[:, 0]
        var_logit = F.softplus(opt[:, 1]) + 1e-8
        
        mu_N = torch.sigmoid(mu_logit)
        jacobian = mu_N * (1.0 - mu_N)
        var_N = (jacobian ** 2) * var_logit + 1e-20

        if self.training:
            return mu_N, var_N
        else:
            return mu_N, var_N, mu_logit, var_logit
       
    
    def act(self, x):
        return F.softplus(x)