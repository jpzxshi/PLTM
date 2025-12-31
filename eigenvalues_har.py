"""
@author: jpzxshi
"""
import numpy as np
import torch
import learner as ln

#### eigenvalue problem of the harmonic oscillator
class STM_eigen(ln.nn.Map):
    '''Spectral-Tensor Model. (L_n+2 - L_n)
    [-5,5]^d, zero bound
    '''
    def __init__(self, dim, interval, bases, rank):
        super(STM_eigen, self).__init__()
        self.dim = dim
        self.interval = interval
        self.bases = bases
        self.rank = rank
        
        self.I_storage = None
        self.dI_storage = None
        self.mI_storage = None
        self.xI_storage = None
        self.pI_storage = None

        self.ps = self.__init_parameters()
        #self.ms = self.__init_modules()
        
    def forward(self, x):
        u = self.ps['coe']
        r = self.func(u, x)
        return r
    
    def func(self, param, x):
        bases = param.size(1)
        x = (x - self.interval[0]) * (2 / (self.interval[1] - self.interval[0])) - 1
        L = [torch.ones_like(x), x]
        for i in range(2, bases + 2):
            L.append(((2 * i - 1) * x * L[-1] - (i - 1) * L[-2]) / i)
        value = torch.stack(L)
        #### L_n+2 - L_n
        value = value[2:] - value[:-2]  # [bases, batch, dim]
        eq = 'ijk,jli->ilk' if len(x.size()) == 2 else 'ijk,ji->ik'
        y = torch.prod(torch.einsum(eq, param, value), dim=0)
        return torch.sum(y, dim=-1, keepdim=True)
        
    def lossfunc(self):
        u = self.ps['coe']
        t, s, r = self.L2_grad(u, u), self.L2(u, u), self.X2(u, u)
        #print('L2', s.item())
        #print('L2_grad', t.item())
        #print('X2', r.item())
        return (t + r) / s
        #return t / s
    
    @property
    def I(self): # I = <P_i, P_j>, [b, b]
        if self.I_storage is not None:
            return self.I_storage
        bases = max([self.bases])
        I1 = np.eye(bases) * (2 / (2 * np.arange(bases) + 1))
        I2 = np.eye(bases) * (2 / (2 * np.arange(bases) + 5))
        I3 = np.eye(bases, k=2) * (2 / (2 * np.arange(bases) + 1))
        I4 = np.eye(bases, k=-2) * (2 / (2 * np.arange(bases) + 1))[:, None]
        
        y = (I1 + I2 - I3 - I4) * ((self.interval[1] - self.interval[0]) / 2)
        self.I_storage = torch.tensor(y, dtype=self.dtype, device=self.device)
        return self.I_storage
    
    @property
    def dI(self): # dI = <P_i',P_j'>, [b, b]
        if self.dI_storage is not None:
            return self.dI_storage
        bases = max([self.bases])
        y = np.diag((np.arange(bases) * 4 + 6) * (2 / (self.interval[1] - self.interval[0])))
        self.dI_storage = torch.tensor(y, dtype=self.dtype, device=self.device)
        return self.dI_storage
    
    @property
    def mI(self): # mI, [d, d, b, b]. mI[i, j] = I if i != j else dI.
        if self.mI_storage is not None:
            return self.mI_storage
        y = []
        for i in range(self.dim):
            y.append(torch.stack([self.I] * i + [self.dI] + [self.I] * (self.dim - i - 1)))
        self.mI_storage = torch.stack(y)
        return self.mI_storage
    
    @property
    def xI(self): # xI = int_(x ** 2 * P_i * P_j), [b, b]
        if self.xI_storage is not None:
            return self.xI_storage
        bases = max([self.bases])
        # L
        L = np.zeros([bases + 2, bases + 2])
        L[0, 0], L[1, 1] = 1, 1
        for i in range(2, bases + 2):
            L[i, :] = (2 * i - 1) / i * np.hstack([np.array([0]), L[i - 1, :-1]]) - (i - 1) / i * L[i - 2, :]
        # P
        P = L[2:] - L[:-2]
        # int_(x ** 2 * P_i * P_j)
        y = np.zeros([bases, bases])
        for i in range(bases):
            for j in range(bases):
                if j < i:
                    y[i, j] = y[j, i]
                else:
                    x2 = np.array([(self.interval[1] + self.interval[0]) / 2, (self.interval[1] - self.interval[0]) / 2])
                    x2 = (self.interval[1] - self.interval[0]) / 2 * np.convolve(x2, x2)
                    p = np.convolve(x2, np.convolve(P[i], P[j]))
                    k = 2 / np.arange(1, 2 * bases + 6)
                    k[1::2] = 0
                    y[i, j] = np.sum(p * k)
        self.xI_storage = torch.tensor(y, dtype=self.dtype, device=self.device)
        return self.xI_storage
    
    @property
    def pI(self): # pI, [d, d, b, b]. pI[i, j] = I if i != j else xI.
        if self.pI_storage is not None:
            return self.pI_storage
        y = []
        for i in range(self.dim):
            y.append(torch.stack([self.I] * i + [self.xI] + [self.I] * (self.dim - i - 1)))
        self.pI_storage = torch.stack(y)
        return self.pI_storage
    
    def L2(self, v1, v2):
        b1, b2 = v1.size(1), v2.size(1)
        return torch.sum(torch.prod(torch.einsum('ijk,ilm,jl->ikm', v1, v2, self.I[:b1, :b2]), dim=0))
    
    def L2_grad(self, v1, v2):
        b1, b2 = v1.size(1), v2.size(1)
        return torch.sum(torch.prod(torch.einsum('ijk,ilm,nijl->nikm', v1, v2, self.mI[..., :b1, :b2]), dim=1))
    
    def H1(self, v1, v2):
        return self.L2(v1, v2) + self.L2_grad(v1, v2)
    
    def X2(self, v1, v2):
        b1, b2 = v1.size(1), v2.size(1)
        return torch.sum(torch.prod(torch.einsum('ijk,ilm,nijl->nikm', v1, v2, self.pI[..., :b1, :b2]), dim=1))
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        ps = torch.zeros([self.dim, self.bases, self.rank])
        ps[:, 0, :] = 0.3 # 1 | 0.3 | 1 / 3.1622776601683795
        parameters['coe'] = torch.nn.Parameter(ps)
        return parameters


#############################################################################
        
class DeepRitz(ln.nn.Algorithm):
    '''The Deep Ritz method for eigenvalue problem of Laplace operator.
    '''
    def __init__(self, stm):
        super(DeepRitz, self).__init__()
        self.stm = stm
        
    def criterion(self, X, y):
        return self.stm.lossfunc()
        #return self.stm.int_gradu2() / self.stm.int_u2()
    
    def predict(self, x, returnnp=False):
        return self.stm.predict(x, returnnp)
    
def plot(data, net):
    print('True eigenvalue:')
    print('{}'.format(net.stm.dim))

def main():
    device = 'gpu' # 'cpu' or 'gpu'
    # stm
    dim = 10 # 10 / 512
    interval = [-5, 5]
    bases = 22 # 22
    rank = 10 # 10
    # training
    lr = 1e-3 # 1e-3 / 1e-3
    iterations = 500 # 500 / 500
    print_every = 100  # 100 / 100
    
    data = ln.Data()
    stm = STM_eigen(dim, interval, bases, rank)
    #print(stm.H1(stm.oc_evs[0], stm.oc_evs[1]))
    net = DeepRitz(stm)
    def callback(data, net): 
        print('{:<9}Eigenvalue: {:<25}'.format('',(net.stm.int_gradu2() / net.stm.int_u2()).item()))
    args = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'double',
        'device': device,
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()

    plot(data, ln.Brain.Best_model())
    
if __name__ == '__main__':
    main()