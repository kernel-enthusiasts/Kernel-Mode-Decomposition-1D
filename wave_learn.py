import numpy as np
import math
from numpy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import KMD_lib
np.random.seed(0)


alpha = 25.0
N = 10000
xL = -1
xU = 1
ss = (xU - xL) / (N - 1)
width = 0.002

t_mesh = np.arange(xL, xU + ss, ss)
t_mesh = t_mesh[:N]

omega_mesh = KMD_lib.make_omega_mesh(t_mesh, alpha)

print(omega_mesh)

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
A2 = (2 + 0.8 * t_mesh)# * np.exp(-(np.maximum(x_mesh_full - 0.25, 0) / 0.05) ** 2)
B2 = (2 - 0.8 * t_mesh ** 3)# * np.exp(-(np.maximum(x_mesh_full - 0.25, 0) / 0.05) ** 2)

Theta = np.zeros((3, N))

for t_indx in range(1, N):
    Theta[0, t_indx] = Theta[0, t_indx - 1] + Freq[0, t_indx - 1] * ss
    Theta[1, t_indx] = Theta[1, t_indx - 1] + Freq[1, t_indx - 1] * ss
    Theta[2, t_indx] = Theta[2, t_indx - 1] + Freq[2, t_indx - 1] * ss

Amp = np.zeros((3, N))
Amp[0] = np.sqrt(A0 ** 2 + B0 ** 2)# * A_vanish
Amp[1] = np.sqrt(A1 ** 2 + B1 ** 2)
Amp[2] = np.sqrt(A2 ** 2 + B2 ** 2)

a2 = np.random.randint(2, size=(7, 2)) * np.random.normal(0, 1, size=(7, 2))
a3 = (1 - np.random.randint(2, size=(7, 2))) * np.random.normal(0, 1, size=(7, 2))

a2[0, 0] = 1.0
a2[0, 1] = 0.0
a3[0, 0] = 1.0
a3[0, 1] = 0.0

a2 = a2 * (1 / ((np.arange(7) + 1.0) ** 2).reshape((-1, 1)))
a3 = a3 * (1 / ((np.arange(7) + 1.0) ** 2).reshape((-1, 1)))

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

signals = np.zeros((3, N))

a = np.zeros((15, 2))
for i in range(8):
    a[2 * i, 0] = (8 / math.pi ** 2) / (2 * i + 1) ** 2

signals0 = Amp[0] * KMD_lib.wave(["custom", a], Theta[0])

signals[0] = Amp[0] * KMD_lib.wave(["tri", 0], Theta[0])
signals[1] = Amp[1] * KMD_lib.wave(["custom", a2], Theta[1])
signals[2] = Amp[2] * KMD_lib.wave(["custom", a3], Theta[2])

signal = np.asarray(signals[0] + signals[1] + signals[2])

Comp_data_full, wp = KMD_lib.semimanual_maxpool_peel2(signal, ["unk", 15], alpha, t_mesh, 0.003, 0.1, ref_fin=True)
wpf = wp.f


N1 = 3333
N2 = 6667
amp1e = Comp_data_full[0, N1:N2, 0]
amp2e = Comp_data_full[1, N1:N2, 0]
amp3e = Comp_data_full[2, N1:N2, 0]
theta1e = Comp_data_full[0, N1:N2, 1]
theta2e = Comp_data_full[1, N1:N2, 1]
theta3e = Comp_data_full[2, N1:N2, 1]
v1e = amp1e * KMD_lib.wave(["custom", wpf[0]], theta1e)
v2e = amp2e * KMD_lib.wave(["custom", wpf[1]], theta2e)
v3e = amp3e * KMD_lib.wave(["custom", wpf[2]], theta3e)

sig1 = signals[0][N1:N2]
sig2 = signals[1][N1:N2]
sig3 = signals[2][N1:N2]
amp1 = Amp[0][N1:N2]
amp2 = Amp[1][N1:N2]
amp3 = Amp[2][N1:N2]

deltheta1 = (Comp_data_full[0, N1:N2, 1] - Theta[0][N1:N2] - math.pi) % (2 * math.pi) - math.pi
deltheta2 = (Comp_data_full[1, N1:N2, 1] - Theta[1][N1:N2] - math.pi) % (2 * math.pi) - math.pi
deltheta3 = (Comp_data_full[2, N1:N2, 1] - Theta[2][N1:N2] - math.pi) % (2 * math.pi) - math.pi

print(LA.norm(v1e - signals0[N1:N2]) / LA.norm(signals0[N1:N2]))
print(LA.norm(v1e - sig1) / LA.norm(sig1))
print(LA.norm(v2e - sig2) / LA.norm(sig2))
print(LA.norm(v3e - sig3) / LA.norm(sig3))

print("XXXXXXXXXXXXXXXXX")

print(np.max(np.abs(v1e - sig1)) / np.max(np.abs(sig1)))
print(np.max(np.abs(v2e - sig2)) / np.max(np.abs(sig2)))
print(np.max(np.abs(v3e - sig3)) / np.max(np.abs(sig3)))
print("XXXXXXXXXXXXXXXXX")


print(LA.norm(amp1e - amp1 * (8 / math.pi ** 2)) / LA.norm(amp1 * (8 / math.pi ** 2)))
print(LA.norm(amp2e - amp2) / LA.norm(amp2))
print(LA.norm(amp3e - amp3) / LA.norm(amp3))
print("XXXXXXXXXXXXXXXXX")


print(((ss / 0.6667) ** 0.5) * LA.norm((Comp_data_full[0, N1:N2, 1] - Theta[0][N1:N2] - math.pi) % (2 * math.pi) - math.pi))
print(((ss / 0.6667) ** 0.5) * LA.norm((Comp_data_full[1, N1:N2, 1] - Theta[1][N1:N2] - math.pi) % (2 * math.pi) - math.pi))
print(((ss / 0.6667) ** 0.5) * LA.norm((Comp_data_full[2, N1:N2, 1] - Theta[2][N1:N2] - math.pi) % (2 * math.pi) - math.pi))

print("XXXXXXXXXXXXXXXXX")

t_mesh1 = np.arange((10000)) * 2 * math.pi / 10000
tri_wave = 1 / (8 / math.pi ** 2) * KMD_lib.wave(["tri", 0], t_mesh1)
a /= 8 / math.pi ** 2
print(LA.norm((tri_wave - KMD_lib.wave(["custom", wpf[0]], t_mesh1))) / LA.norm(tri_wave))
print(LA.norm((KMD_lib.wave(["custom", a], t_mesh1) - KMD_lib.wave(["custom", wpf[0]], t_mesh1))) / LA.norm(KMD_lib.wave(["custom", a], t_mesh1)))
print(LA.norm((KMD_lib.wave(["custom", a2], t_mesh1) - KMD_lib.wave(["custom", wpf[1]], t_mesh1))) / LA.norm(KMD_lib.wave(["custom", a2], t_mesh1)))
print(LA.norm((KMD_lib.wave(["custom", a3], t_mesh1) - KMD_lib.wave(["custom", wpf[2]], t_mesh1))) / LA.norm(KMD_lib.wave(["custom", a3], t_mesh1)))
