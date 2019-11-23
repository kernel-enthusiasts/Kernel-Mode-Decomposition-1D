import numpy as np
import math
from numpy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import KMD_lib
np.random.seed(0)


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
wave_params = ["tri", 0]

alpha = 25.0
N = 10000
xL = -1
xU = 1
ss = (xU - xL) / (N - 1)
sigma = 0.5

t_mesh = np.arange(xL, xU + ss, ss)
t_mesh = t_mesh[:N]

omega_mesh = KMD_lib.make_omega_mesh(t_mesh, alpha)

omega_N = omega_mesh.shape[0]

T0 = 0.3 / (1 + 0.2 * (t_mesh - 0.5) ** 2)
T1 = (1 / 14) * (1 + 0.2 * (t_mesh + 0.8) ** 2)
T2 = (1 / 20) * (1 - 0.2 * t_mesh - 0.01 * np.abs(t_mesh ** 3))
Freq = np.zeros((3, N))
Freq[0] = 12 * math.pi / T0
Freq[1] = 16 * math.pi / T1
Freq[2] = 9 * math.pi / T2

A_vanish = np.zeros((N))

for i in range(N - 1):
    A_vanish[i + 1] = A_vanish[i] + math.exp(-((t_mesh[i] + 0.5) / 0.15) ** 2)

A_vanish /= np.max(A_vanish)

A_vanish = 1 - A_vanish


A0 = 2.5 + 0.5 * t_mesh ** 2
B0 = -1.25 * 0.5 * t_mesh
A1 = (2 - 0.8 * t_mesh ** 3)
B1 = (2 - 0.8 * t_mesh + 0.5 * t_mesh ** 2)
A2 = (2 + 0.8 * t_mesh)
B2 = (2 - 0.8 * t_mesh ** 3)

Theta = np.zeros((3, N))

for t_indx in range(1, N):
    Theta[0, t_indx] = Theta[0, t_indx - 1] + Freq[0, t_indx - 1] * ss
    Theta[1, t_indx] = Theta[1, t_indx - 1] + Freq[1, t_indx - 1] * ss
    Theta[2, t_indx] = Theta[2, t_indx - 1] + Freq[2, t_indx - 1] * ss

Amp = np.zeros((3, N))
Amp[0] = np.sqrt(A0 ** 2 + B0 ** 2) * A_vanish
Amp[1] = np.sqrt(A1 ** 2 + B1 ** 2)
Amp[2] = np.sqrt(A2 ** 2 + B2 ** 2)

DTheta0 = np.arctan2(-B0, A0)
DTheta1 = np.arctan2(-B1, A1)
DTheta2 = np.arctan2(-B2, A2)

for T_indx in range(N - 1):
    if DTheta0[T_indx + 1] - DTheta0[T_indx] > math.pi:
        DTheta0[T_indx + 1:] -= 2 * math.pi
    if DTheta0[T_indx + 1] - DTheta0[T_indx] < -math.pi:
        DTheta0[T_indx + 1:] += 2 * math.pi

    if DTheta1[T_indx + 1] - DTheta1[T_indx] > math.pi:
        DTheta1[T_indx + 1:] -= 2 * math.pi
    if DTheta1[T_indx + 1] - DTheta1[T_indx] < -math.pi:
        DTheta1[T_indx + 1:] += 2 * math.pi

    if DTheta2[T_indx + 1] - DTheta2[T_indx] > math.pi:
        DTheta2[T_indx + 1:] -= 2 * math.pi
    if DTheta2[T_indx + 1] - DTheta2[T_indx] < -math.pi:
        DTheta2[T_indx + 1:] += 2 * math.pi

Theta[0] += DTheta0
Theta[1] += DTheta1
Theta[2] += DTheta2

signals = np.zeros((4, N))

signals[0] = Amp[0] * KMD_lib.wave(wave_params, Theta[0])
signals[1] = Amp[1] * KMD_lib.wave(wave_params, Theta[1])
signals[2] = Amp[2] * KMD_lib.wave(wave_params, Theta[2])
signals[3] = np.random.normal(0, 0.5, size=(N))

#prints the SNR
#print(np.mean(signals[3] ** 2))
#print(np.mean(signals[0] ** 2) / np.mean(signals[3] ** 2))
#print(np.mean(signals[1] ** 2) / np.mean(signals[3] ** 2))
#print(np.mean(signals[2, :3000] ** 2) / np.mean(signals[3, :3000] ** 2))
#print(np.mean((signals[0] + signals[1] + signals[2]) ** 2) / np.mean(signals[3] ** 2))

signal = np.asarray(signals[0] + signals[1] + signals[2] + signals[3])

Comp_data_full, wp = KMD_lib.semimanual_maxpool_peel2(signal, wave_params, alpha, t_mesh, 0.005, 0.1, ref_fin=False)


plt.plot(t_mesh, Comp_data_full[0, :, 0] * KMD_lib.wave(wave_params, Comp_data_full[0, :, 1]), label=r"$v_{1,e}$")
plt.plot(t_mesh, signals[0], label=r"$v_1$")
plt.legend()
plt.show()

plt.plot(t_mesh, Comp_data_full[2, :, 0] * KMD_lib.wave(wave_params, Comp_data_full[2, :, 1]), label=r"$v_{2,e}$")
plt.plot(t_mesh, signals[1], label=r"$v_2$")
plt.legend()
plt.show()

plt.plot(t_mesh, Comp_data_full[1, :, 0] * KMD_lib.wave(wave_params, Comp_data_full[1, :, 1]), label=r"$v_{3,e}$")
plt.plot(t_mesh, signals[2], label=r"$v_3$")
plt.legend()
plt.show()

