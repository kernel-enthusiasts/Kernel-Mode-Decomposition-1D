import numpy as np
import math
from numpy import linalg as LA
import matplotlib
#matplotlib.use("nbagg")
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import KMD_lib
np.random.seed(0)

wave_params = ["ekg", 0]

#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 26}
#
#matplotlib.rc('font', **font)

alpha = 25.0
N = 10000
xL = -1
xU = 1
ss = (xU - xL) / (N - 1)

t_mesh = np.arange(xL, xU + ss, ss)
t_mesh = t_mesh[:N]


omega_mesh = KMD_lib.make_omega_mesh(t_mesh, alpha)

A_vanish = np.zeros((N))

for i in range(N - 1):
    A_vanish[i + 1] = A_vanish[i] + math.exp(-((t_mesh[i] + 0.4) / 0.25) ** 2)

A_vanish /= np.max(A_vanish)

T0 = 0.3 / (1 + 0.2 * (t_mesh + 0.5) ** 2)
T1 = (1 / 14) * (1 + 0.2 * (t_mesh + 0.5) ** 2)
T2 = (1 / 20) * (1 + 0.2 * t_mesh + 0.01 * np.abs(t_mesh ** 3))
Freq = np.zeros((3, N))
Freq[0] = 12 * math.pi / T0
Freq[1] = 12 * math.pi / T1
Freq[2] = 12 * math.pi / T2

A0 = 1 + 0.2 * t_mesh ** 2
B0 = -0.5 * 0.2 * t_mesh
A1 = (2 + 0.8 * t_mesh ** 3)
B1 = (2 + 0.8 * t_mesh + 0.5 * t_mesh ** 2)
A2 = (2 + 0.8 * t_mesh)
B2 = (2 - 0.8 * t_mesh ** 3)

Theta = np.zeros((3, N))


for t_indx in range(1, N):
    Theta[0, t_indx] = Theta[0, t_indx - 1] + Freq[0, t_indx - 1] * ss
    Theta[1, t_indx] = Theta[1, t_indx - 1] + Freq[1, t_indx - 1] * ss
    Theta[2, t_indx] = Theta[2, t_indx - 1] + Freq[2, t_indx - 1] * ss

Amp = np.zeros((3, N))
Amp[0] = np.sqrt(A0 ** 2 + B0 ** 2)# * A_vanish
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
signals[3] = np.random.normal(0, 0.0, size=(N))

signal = np.asarray(signals[0] + signals[1] + signals[2] + signals[3])


#Comp_data_full, wp = KMD_lib.semimanual_maxpool_peel2(signal, ["unk", 15], alpha, t_mesh, 0.003, 0.05, ref_fin=False)
#wave_params1 = ["custom", wp.f[0]]
#wave_params2 = ["custom", wp.f[1]]
#wave_params3 = ["custom", wp.f[2]]

Comp_data_full, wp = KMD_lib.semimanual_maxpool_peel2(signal, wave_params, alpha, t_mesh, 0.03, 0.05, ref_fin=True)

for i in range(3):
    plt.plot(t_mesh, Comp_data_full[i, :, 0] * KMD_lib.wave(wave_params, Comp_data_full[i, :, 1]))
    plt.plot(t_mesh, signals[i])
    plt.show()

    plt.plot(t_mesh, Comp_data_full[i, :, 0] * KMD_lib.wave(wave_params, Comp_data_full[i, :, 1]) - signals[i])
    plt.show()


N1 = 3333
N2 = 6667
amp1e = Comp_data_full[0, N1:N2, 0]
amp2e = Comp_data_full[1, N1:N2, 0]
amp3e = Comp_data_full[2, N1:N2, 0]
theta1e = Comp_data_full[0, N1:N2, 1]
theta2e = Comp_data_full[1, N1:N2, 1]
theta3e = Comp_data_full[2, N1:N2, 1]
v1e = amp1e * KMD_lib.wave(wave_params, theta1e)
v2e = amp2e * KMD_lib.wave(wave_params, theta2e)
v3e = amp3e * KMD_lib.wave(wave_params, theta3e)

sig1 = signals[0][N1:N2]
sig2 = signals[1][N1:N2]
sig3 = signals[2][N1:N2]
amp1 = Amp[0][N1:N2]
amp2 = Amp[1][N1:N2]
amp3 = Amp[2][N1:N2]

deltheta1 = (Comp_data_full[0, N1:N2, 1] - Theta[0][N1:N2] - math.pi) % (2 * math.pi) - math.pi
deltheta2 = (Comp_data_full[1, N1:N2, 1] - Theta[1][N1:N2] - math.pi) % (2 * math.pi) - math.pi
deltheta3 = (Comp_data_full[2, N1:N2, 1] - Theta[2][N1:N2] - math.pi) % (2 * math.pi) - math.pi

print(LA.norm(v1e - sig1) / LA.norm(sig1))
print(LA.norm(v2e - sig2) / LA.norm(sig2))
print(LA.norm(v3e - sig3) / LA.norm(sig3))

print("XXXXXXXXXXXXXXXXX")

print(np.max(np.abs(v1e - sig1)) / np.max(np.abs(sig1)))
print(np.max(np.abs(v2e - sig2)) / np.max(np.abs(sig2)))
print(np.max(np.abs(v3e - sig3)) / np.max(np.abs(sig3)))
print("XXXXXXXXXXXXXXXXX")


print(LA.norm(amp1e - amp1) / LA.norm(amp1))
print(LA.norm(amp2e - amp2) / LA.norm(amp2))
print(LA.norm(amp3e - amp3) / LA.norm(amp3))
print("XXXXXXXXXXXXXXXXX")


print(((ss / 0.6667) ** 0.5) * LA.norm((Comp_data_full[0, N1:N2, 1] - Theta[0][N1:N2] - math.pi) % (2 * math.pi) - math.pi))
print(((ss / 0.6667) ** 0.5) * LA.norm((Comp_data_full[1, N1:N2, 1] - Theta[1][N1:N2] - math.pi) % (2 * math.pi) - math.pi))
print(((ss / 0.6667) ** 0.5) * LA.norm((Comp_data_full[2, N1:N2, 1] - Theta[2][N1:N2] - math.pi) % (2 * math.pi) - math.pi))

print("XXXXXXXXXXXXXXXXX")

