import numpy as np
import matplotlib.pylab as plt

class DiscreteSC:
    def __init__(self, n_dim, n_sparse, eta_A, eta_s, n_batch, l0, l1, sigma):
        self.n_dim = n_dim
        self.n_sparse = n_sparse
        self.eta_A = eta_A
        self.eta_s = eta_s
        self.n_batch = n_batch

        self.l0 = l0
        self.l1 = l1
        self.sigma = sigma

        self.s0 = -np.log(1 - l0) / l1
    def dEds(self, s, x, A):
        sign_s = np.sign(s)
        where_active = (np.abs(s) >= self.s0)
        u = (s - self.s0*(sign_s)) * where_active
        dE_recon = (A.T @ (A @ u - x) / self.sigma**2) * where_active
        dE_sparse = self.l1 * sign_s
        #dE_sparse = self.l1 * ( (s > 0) - 10*(s < 0) )
        return dE_recon + dE_sparse
    def dEdA(self, s, x, A):
        u = (s - self.s0*(np.sign(s))) * (np.abs(s) > self.s0)
        return (A @ u - x)[:, None] @ u[None, :] / self.sigma**2
    def init_loader(self, X):
        self.X = np.random.permutation(X)
        self.batch_idx = 0
    def get_batch(self):
        batch_end = self.batch_idx + self.n_batch
        batch = self.X[self.batch_idx:batch_end]

        if batch_end >= len(self.X):
            batch_end %= len(self.X)
            self.init_loader(self.X)
            batch = np.vstack((batch, self.X[:batch_end]))
        self.batch_idx = batch_end
        return batch
    def init_A(self):
        A = np.random.normal(0, 0.4, size=(self.n_dim, self.n_sparse))
        #A = np.eye(2)
        return A
    def init_s(self):
        s = np.zeros(self.n_sparse)
        #s = np.array((1, 0.))
        return s
    def solve(self, X, n_iter, max_iter_s=100, eps=1e-5):
        A = self.init_A()

        A_soln = np.zeros((n_iter, self.n_dim, self.n_sparse))
        s_soln = np.zeros((n_iter, self.n_batch, self.n_sparse))
        X_soln = np.zeros((n_iter, self.n_batch, self.n_dim))

        # Init batch counter
        self.init_loader(X)
        # A update loop
        for n in range(n_iter):
            X_batch = self.get_batch()
            X_soln[n] = X_batch

            s = self.init_s()
            # Data loop
            for n_x, x in enumerate(X_batch):
                ds = 0
                # Find MAP of s
                for i in range(max_iter_s):
                    ds = - self.eta_s * self.dEds(s, x, A)
                    if (ds**2).mean()**0.5 < eps:
                        break
                    s += ds
                s_soln[n, n_x] = s
                A += (-self.eta_A * self.dEdA(s, x, A))/self.n_batch
            # Normalize A
            A /= np.linalg.norm(A, axis=0)
            A_soln[n] = A
        solns = {
                'A' : A_soln,
                's' : s_soln,
                'x' : X_soln
                }
        return solns

if __name__ == '__main__':
    n_dim = 2
    n_sparse = 2
    n_batch = 2
    eta_A = 1e-1
    eta_s = 1e-1
    l0 = .0
    l1 = .5
    sigma = 1

    dsc = DiscreteSC(n_dim, n_sparse, eta_A, eta_s, n_batch, l0, l1, sigma)

    theta = (np.linspace(0, 2*np.pi, n_sparse, endpoint=False))
    cos = np.cos(theta)
    sin = np.sin(theta)
    X = np.hstack((cos[:,None], sin[:,None]))
    X *= 10
    #X = np.zeros((1, 2))

    solns = dsc.solve(X, n_iter=10)
    print(solns['s'])
