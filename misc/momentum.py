import numpy as np
import tqdm
import matplotlib.pylab as plt

def dEdx(x):
    return x

def update_param(p, dEdx, dt, tau, mu, T=1, dW=None):
    if dW is None:
        T = 0
    if mu == 0:
        # No mass
        dx = -dEdx / tau
        if T > 0:
            dx += (T / tau)**0.5 * dW
        return dx, 0
    else:
        # Mass
        m = mu * tau**2
        dx = p * dt / (2*m)
        dp = -tau * p * dt / m - dEdx
        if T > 0:
            dp += (T * tau)**0.5 * dW
        dx += p * dt / (2*m)
        return dx, dp
def solve_w_momentum(tspan, x0=0, m=1, gamma=1, T=1):
    dT = tspan[1:] - tspan[:-1]
    dW = np.random.normal(0, scale = (2*dT)**0.5)
    x = x0
    p = 0
    Xs = [x]
    Ps = [p]
    for i, dt in enumerate(dT):
        dx, dp = update_param(p, dEdx(x), dt, gamma, m, T, dW[i])
        x += dx
        p += dp
        Xs.append(x)
        Ps.append(p)
    return Xs, Ps

T_STEPS = 10000
tspan = np.linspace(0, T_STEPS, T_STEPS)
x0 = 10
gamma = 10
m = 10
T = 1
Xs, Ps = solve_w_momentum(tspan, x0, m, gamma, T)


xspan = np.linspace(-10, 10, 100)
dx = xspan[1] - xspan[0]
f = np.exp(-.5*xspan**2)
f /= f.sum() * dx
plt.subplot(311)
plt.plot(Xs)
plt.subplot(312)
plt.hist(Xs, bins=100, density=True)
plt.plot(xspan, f)
plt.subplot(313)
plt.plot(np.correlate(Xs, Xs, mode='full'))
plt.show()
