from ctsc import *
import numpy as np
import torch as th
import h5py
import matplotlib.pylab as plt

class SolnAnalysis:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.soln = h5py.File(os.path.join(dir_path, 'soln.h5'))
        self.solver = load_solver(dir_path)
        self.model = self.solver.model
        self.t_steps = len(self.soln['t'])
    def energy(self):
        X_soln = self.soln['x']
        U_soln = self.soln['u']
        A_soln = self.soln['A']
        energy = np.zeros(self.t_steps)
        recon = np.zeros(self.t_steps)
        for n, (A, u, x) in enumerate(zip(A_soln, U_soln, X_soln)):
            energy[n], recon[n] = self.model.energy(x, A=A, u=u, return_recon=True)


        return energy, recon

if __name__ == '__main__':
    dir_path = get_timestamped_dir(load=True, base_dir='bars_dsc')
    analysis = SolnAnalysis(dir_path)
    #print(analysis.solver.tau_x)
    time = analysis.soln['t'][:]
    energy, recon = analysis.energy()
    plt.plot(analysis.soln['t'], recon, 'g')
    plt.yscale('log')

    dir_path = get_timestamped_dir(load=True, base_dir='bars_0T')
    analysis = SolnAnalysis(dir_path)
    #print(analysis.solver.tau_x)
    time = analysis.soln['t'][:]
    energy, recon = analysis.energy()
    plt.plot(analysis.soln['t'], recon, 'r')

    dir_path = get_timestamped_dir(load=True, base_dir='bars_asynch')
    analysis = SolnAnalysis(dir_path)
    #print(analysis.solver.tau_x)
    time = analysis.soln['t'][:]
    energy, recon = analysis.energy()
    plt.plot(analysis.soln['t'], recon, 'b')

    plt.yscale('log')
    plt.show()
