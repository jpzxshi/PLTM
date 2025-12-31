"""
@author: jpzxshi
"""
import numpy as np
import torch
import learner as ln

#### eigenvalue problem of laplace operator
class STM_eigen(ln.nn.Map):
    '''Spectral-Tensor Model. (L_n+2 - L_n)
    [0,1]^d, zero bound
    '''
    def __init__(self, dim, interval, bases, rank, oc_evs=[]):
        super(STM_eigen, self).__init__()
        self.dim = dim
        self.interval = interval
        self.bases = bases
        self.rank = rank
        self.oc_evs = oc_evs # list of [d, b, r] tensors
        
        self.I_storage = None
        self.dI_storage = None
        self.mI_storage = None

        self.ps = self.__init_parameters()
        #self.ms = self.__init_modules()
        
    def forward(self, x):
        u = self.ps['coe']
        r = self.func(u, x)
        for ui in self.oc_evs:
            a = self.H1(u, ui) / self.H1(ui, ui)
            r = r - a * self.func(ui, x)
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
    
    def get_func_param(self):
        u = self.ps['coe'].detach()
        #### modify
        evs = []
        a = []
        for ui in self.oc_evs:
            a.append(self.H1(u, ui) / self.H1(ui, ui))
        b_max = max([u.size(1)] + [ui.size(1) for ui in self.oc_evs])
        if u.size(1) < b_max:
            zeros = torch.zeros([u.size(0), b_max - u.size(1), u.size(2)], dtype=u.dtype, device=u.device)
            u = torch.cat([u, zeros], dim=1)
        for i in range(len(self.oc_evs)):
            ui = torch.pow(a[i], 1 / self.dim) * self.oc_evs[i]
            ui[0, :, :] = - ui[0, :, :]
            if ui.size(1) < b_max:
                zeros = torch.zeros([ui.size(0), b_max - ui.size(1), ui.size(2)], dtype=ui.dtype, device=ui.device)
                ui = torch.cat([ui, zeros], dim=1)
            evs.append(ui)
        ####
        return torch.cat([u] + evs, dim=2)
        
    def lossfunc(self):
        u = self.ps['coe']
        t, s = self.L2_grad(u, u), self.L2(u, u)
        #s, t = self.L2(u, u), self.L2_grad(u, u)
        for ui in self.oc_evs:
            a = self.H1(u, ui) / self.H1(ui, ui)
            #a = self.L2(u, ui) / self.L2(ui, ui)
            s = s + a ** 2 * self.L2(ui, ui) - 2 * a * self.L2(u, ui)
            t = t + a ** 2 * self.L2_grad(ui, ui) - 2 * a * self.L2_grad(u, ui)
        return t / s
    
    @property
    def I(self): # I = <P_i, P_j>, [b, b]
        if self.I_storage is not None:
            return self.I_storage
        bases = max([self.bases] + [t.size(1) for t in self.oc_evs])
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
        bases = max([self.bases] + [t.size(1) for t in self.oc_evs])
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
    
    def L2(self, v1, v2):
        b1, b2 = v1.size(1), v2.size(1)
        return torch.sum(torch.prod(torch.einsum('ijk,ilm,jl->ikm', v1, v2, self.I[:b1, :b2]), dim=0))
    
    def L2_grad(self, v1, v2):
        b1, b2 = v1.size(1), v2.size(1)
        return torch.sum(torch.prod(torch.einsum('ijk,ilm,nijl->nikm', v1, v2, self.mI[..., :b1, :b2]), dim=1))
    
    def H1(self, v1, v2):
        return self.L2(v1, v2) + self.L2_grad(v1, v2)
    
    def __init_parameters(self):
        parameters = torch.nn.ParameterDict()
        #ps = torch.zeros([self.dim, self.bases, self.rank])
        ps = torch.rand([self.dim, self.bases, self.rank]) * 1e-5
        ps[:, 0, :] = 1
        parameters['coe'] = torch.nn.Parameter(ps)
        return parameters
        
def test():
    l98 = torch.load('outputs/d10lam98/model_best.pkl').stm
    l177 = torch.load('outputs/d10lam177/model_best.pkl').stm
    l177_2 = torch.load('outputs/d10lam177_2/model_best.pkl').stm
    l177_3 = torch.load('outputs/d10lam177_3/model_best.pkl').stm
    l177_4 = torch.load('outputs/d10lam177_4/model_best.pkl').stm
    
    print(l98.lossfunc().item())
    print()

    print(l177.get_func_param().size())
    print(l177.H1(l98.get_func_param(), l98.get_func_param()))
    print(l177.H1(l177.get_func_param(), l177.get_func_param()))
    print(l177.H1(l177.get_func_param(), l98.get_func_param()))
    print(l177.lossfunc().item())
    print()
    
    print(l177_2.get_func_param().size())
    print(l177_2.H1(l177_2.get_func_param(), l177_2.get_func_param()))
    print(l177_2.H1(l177_2.get_func_param(), l98.get_func_param()))
    print(l177_2.H1(l177_2.get_func_param(), l177.get_func_param()))
    print(l177_2.lossfunc().item())
    print()
    
    print(l177_3.get_func_param().size())
    print(l177_3.H1(l177_3.get_func_param(), l177_3.get_func_param()))
    print(l177_3.H1(l177_3.get_func_param(), l98.get_func_param()))
    print(l177_3.H1(l177_3.get_func_param(), l177.get_func_param()))
    print(l177_3.H1(l177_3.get_func_param(), l177_2.get_func_param()))
    print(l177_3.lossfunc().item())
    print()

    print(l177_4.get_func_param().size())
    print(l177_4.H1(l177_4.get_func_param(), l177_4.get_func_param()))
    print(l177_4.H1(l177_4.get_func_param(), l98.get_func_param()))
    print(l177_4.H1(l177_4.get_func_param(), l177.get_func_param()))
    print(l177_4.H1(l177_4.get_func_param(), l177_2.get_func_param()))
    print(l177_4.H1(l177_4.get_func_param(), l177_3.get_func_param()))
    print(l177_4.lossfunc().item())


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
    print('{}*pi^2={}'.format(net.stm.dim, net.stm.dim * np.pi ** 2))
    #test()
    # d=1  9.869604401089358
    # d=5  49.34802200544679
    # d=10 98.69604401089359   10 
    #      128.30485721416164  13 2
    #      157.91367041742973  16 2 2
    #      177.65287921960845  18 3
    #      187.5224836206978   19 2 2 2
    #      207.2616924228765   21 3 2
    #      217.13129682396587  22 2 2 2 2
    #      236.8705056261446   24 3 2 2
    #      246.74011002723395  25 4 / 2 2 2 2 2

def main():
    device = 'gpu' # 'cpu' or 'gpu'
    # stm
    dim = 10 # 10 / 512
    interval = [0, 1]
    bases = 10 # 10 / 10
    rank = 10  # 10 / 10
    # training
    lr = 1e-3 # 1e-3 / 1e-3
    iterations = 500 # 500 / 500
    print_every = 100  # 100 / 100
    oc_evs = []#torch.load('outputs/d10lam98/model_best.pkl').stm.get_func_param(),
              #torch.load('outputs/d10lam177/model_best.pkl').stm.get_func_param(),
              #torch.load('outputs/d10lam177_2/model_best.pkl').stm.get_func_param(),
              #torch.load('outputs/d10lam177_3/model_best.pkl').stm.get_func_param()]
    
    data = ln.Data()
    stm = STM_eigen(dim, interval, bases, rank, oc_evs=oc_evs)
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