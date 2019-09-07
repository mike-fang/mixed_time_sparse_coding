import torch as th
from torch.optim.optimizer import Optimizer, required


class EulerMaruyama(Optimizer):
    def __init__(self,  params, dt=required, tau=1, mu=0, T=0):
        defaults = dict(dt=dt, tau=tau, mu=mu, T=T)
        super().__init__(params, defaults)
    def step(self, closure):
        # Half-step x if there is mass
        for group in self.param_groups:
            mu = group['mu']
            if mu == 0:
                continue
            tau = group['tau']
            du =  group['dt'] / tau

            for p in group['params']:
                param_state = self.state[p]
                if 'momentum' not in param_state:
                    param_state['momentum'] = th.zeros_like(p)
                pi = param_state['momentum']
                p.data.add_(du/(2 * mu), pi)

        energy = closure() 

        for group in self.param_groups:
            mu = group['mu']
            tau = group['tau']
            T = group['T']
            du =  group['dt'] / tau

            for p in group['params']:
                p_grad = p.grad
                eta = th.FloatTensor(p.shape).normal_()

                if mu > 0:
                    # Step momentum if there is mass
                    pi = self.state[p]['momentum']
                    pi.add_(-du, pi/mu + p_grad)
                    
                    # Add noise if there is temperature
                    if T > 0:
                        pi.add_((2 * T * du)**0.5, eta)

                    # Half-step x
                    p.data.add_(du/(2 * mu), pi)
                else:
                    # If no mass, just step x
                    p.data.add_(-du, p_grad)
                    if T > 0:
                        p.data.add_((2 * T * du)**0.5, eta)
        return energy

if __name__ == '__main__':
    from torch.nn import Parameter
    import matplotlib.pylab as plt
    import numpy as np

    x = Parameter(th.tensor(0.))
    def closure():
        energy = (x**2)/2
        energy.backward()
        return energy

    param_groups = [
            {'params': [x], 'tau': 1e1, 'mu': 10, 'T': 1}
            ]
    optimizer = EulerMaruyama(param_groups, dt=1)
    X = []
    for _ in range(10000):
        optimizer.zero_grad()
        optimizer.step(closure)
        X.append(float(x))

    plt.subplot(211)
    plt.plot(X)
    plt.subplot(212)
    X_range = np.linspace(-3, 3, 100)
    p = np.exp(-X_range**2/2)
    p /= p.sum() * (X_range[1] - X_range[0])
    
    plt.hist(X, bins=50, density=True)
    plt.plot(X_range, p)
    plt.show()
