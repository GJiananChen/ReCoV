import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
import numpy as np

class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class
        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.
        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
        self.reg = Regularization(2, weight_decay=0)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0]).cuda()
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        # l2_loss = self.reg(model)
        l2_loss = 0
        return neg_log_loss + l2_loss

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index
    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

class MINN(nn.Module):
    def __init__(self, pooling='mean', activation='relu'):
        super(MINN, self).__init__()
        self.pooling = pooling
        self.activation = activation
        self.select_activation()

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(103, 64),
            self.act,
            nn.Linear(64, 32),
            self.act,
            nn.Linear(32, 32),
            self.act,
            nn.Linear(32, 32),
            self.act,
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 1),
            nn.SELU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def select_activation(self):
        if self.activation == 'SELU':
            self.act = nn.SELU()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = x.squeeze(0)
        n = x.shape[0]
        H = self.feature_extractor_part1(x)  # NxL

        if self.pooling == 'max':
            H_B, _ = torch.max(H, 0)
        elif self.pooling == 'mean':
            H_B = torch.mean(H, 0)
        elif self.pooling == 'sum':
            H_B = torch.sum(H, 0)
        elif self.pooling == 'att':
            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N
            H_B = torch.mm(A, H)  # KxL
        elif self.pooling == 'satt':
            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = torch.sigmoid(A)  # softmax over N
            A = torch.mul(A, n)
            H_B = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(H_B)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        A = 0
        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat, Y_prob

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (
                    Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

# current best model
# tried batchnorm but it's normalizing instances instead of batches
class MIL_reg_Ins(nn.Module):
    def __init__(self, pooling='mean', n_input=128, activation='ReLU'):
        super(MIL_reg_Ins, self).__init__()
        self.pooling = pooling
        if activation == 'SELU':
            act_fun = nn.SELU()
        else:
            act_fun = nn.ReLU()
        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(n_input, 128),
            torch.nn.LayerNorm([128]),
            # torch.nn.BatchNorm1d(128),
            act_fun,
            nn.Linear(128, 64),
            torch.nn.LayerNorm([64]),
            # torch.nn.BatchNorm1d(64),
            act_fun,

            nn.Linear(64, 32),
            torch.nn.LayerNorm([32]),
            # torch.nn.BatchNorm1d(32),
            act_fun,
            nn.Linear(32, 32),
            torch.nn.LayerNorm([32]),
            # torch.nn.BatchNorm1d(32),
            act_fun,
            nn.Linear(32, 32),
            torch.nn.LayerNorm([32]),
            # torch.nn.BatchNorm1d(32),
            act_fun,
        )

        self.classifier = nn.Sequential(
            nn.Linear(32,1),
            nn.SELU()
        )

        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        n = x.shape[0]
        x = x.squeeze(0)
        if len(x.shape)==1:
            x = torch.unsqueeze(x, 0)
        H = self.feature_extractor_part1(x)  # NxL
        Y_ins = self.classifier(H)
        # Y_ins = torch.exp(Y_ins)
        if n == 1:
            Y = Y_ins
        else:
            if self.pooling == 'max':
                Y, _ = torch.max(Y_ins, 0)
            elif self.pooling == 'mean':
                Y = torch.mean(Y_ins, 0)
            elif self.pooling == 'sum':
                Y = torch.sum(Y_ins, 0)
            elif self.pooling == 'att':
                Y = self.attention(Y_ins)  # NxK
                Y = torch.transpose(Y, 1, 0)  # KxN
                Y = F.softmax(Y, dim=1)  # softmax over N
                Y = torch.mm(Y, H)  # KxL
            elif self.pooling == 'satt':
                Y = self.attention(Y_ins, n)  # NxK
                Y = torch.transpose(Y, 1, 0)  # KxN
                Y = torch.sigmoid(Y)  # softmax over N
                Y = torch.mul(Y, n)
                Y = torch.mm(Y, H)  # KxL
        # H_B = torch.sum(H, 0)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        A = 0
        # print(Y)
        # print(Y_ins)
        return Y, Y_ins, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_hat, Y_ins, _ = self.forward(X)
        if Y_hat >= 0.5:
            Y_pred = 1
        else:
            Y_pred = 0
        error = 1- int(Y.cpu().mean().data.item() == Y_pred)
        return error, Y_ins, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class MIL_reg_Ins_cli(nn.Module):
    def __init__(self, pooling='mean', activation='SELU'):
        super(MIL_reg_Ins, self).__init__()
        self.pooling = pooling

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(103, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32,1),
            nn.SELU()
        )

        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        if len(x.shape)==1:
            x = torch.unsqueeze(x, 0)
        n = x.shape[0]
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)  # NxL
        Y_ins = self.classifier(H)
        # Y_ins = torch.exp(Y_ins)
        if n == 1:
            Y = Y_ins
        else:
            if self.pooling == 'max':
                Y, _ = torch.max(Y_ins, 0)
            elif self.pooling == 'mean':
                Y = torch.mean(Y_ins, 0)
            elif self.pooling == 'sum':
                Y = torch.sum(Y_ins, 0)
            elif self.pooling == 'att':
                Y = self.attention(Y_ins)  # NxK
                Y = torch.transpose(Y, 1, 0)  # KxN
                Y = F.softmax(Y, dim=1)  # softmax over N
                Y = torch.mm(Y, H)  # KxL
            elif self.pooling == 'satt':
                Y = self.attention(Y_ins, n)  # NxK
                Y = torch.transpose(Y, 1, 0)  # KxN
                Y = torch.sigmoid(Y)  # softmax over N
                Y = torch.mul(Y, n)
                Y = torch.mm(Y, H)  # KxL
        # H_B = torch.sum(H, 0)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        A = 0
        # print(Y)
        # print(Y_ins)
        return Y, Y_ins, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        return error, Y_hat, Y_prob

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A