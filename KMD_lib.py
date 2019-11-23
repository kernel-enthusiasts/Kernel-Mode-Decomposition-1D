import numpy as np
import math
from numpy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import time
import scipy.signal as scisig
import copy
import scipy.optimize
from scipy.interpolate import CubicSpline


def wave(wave_params, x):
    #Constructs and returns a wave
    #wave_params[0] has base waveform
    #wave_params[1] has wave parameters (only req for custom) and wave_params[1][0, 1] = 0
    #x is the time mesh
    if wave_params[0] == "cos":
        return np.cos(x)
    if wave_params[0] == "tri":
        return tri_wave(x)
    if wave_params[0] == "ekg":
        return ekg_wave(x)
    if wave_params[0] == "custom":
        c = wave_params[1]
        wave = np.zeros_like(x)
        for i in range(c.shape[0]):
            wave += c[i, 0] * np.cos((i + 1) * x)
            if i > 0: wave += c[i, 1] * np.sin((i + 1) * x)
        return wave

def fund_wave(wave_params, x):
    #fundamental frequency component of wave
    #wave_params[0] has base waveform
    #wave_params[1] has wave parameters (only req for custom) and wave_params[1][0, 1] = 0
    #x is the time mesh

    if wave_params[0] == "cos":
        return np.cos(x)
    if wave_params[0] == "tri":
        return (8 / math.pi ** 2) * np.cos(x)
    if wave_params[0] == "ekg":
        return -0.24495445597949841 * np.cos(x)
    if wave_params[0] == "custom":
        c = wave_params[1]
        wave = c[0, 0] * np.cos(x)
        return wave

def fund_coeff(wave_params):
    #returns fundamental frequency coefficient of base waveform
    #wave_params[0] has base waveform
    #wave_params[1] has wave parameters (only req for custom) and wave_params[1][0, 1] = 0

    if wave_params[0] == "cos":
        return 1
    if wave_params[0] == "tri":
        return (8 / math.pi ** 2)
    if wave_params[0] == "ekg":
        return -0.24495445597949841
    if wave_params[0] == "custom":
        c = wave_params[1]
        return c[0, 0]

def ovtn_wave(wave_params, x):
    #constructs and returns overtone frequencies component of wave
    #wave_params[0] has base waveform
    #wave_params[1] has wave parameters (only req for custom) and wave_params[1][0, 1] = 0
    return wave(wave_params, x) - fund_wave(wave_params, x)

def tri_wave(x):
    #constructs and returns a triangular wave with time mesh x
    #Normalized to have period 2*pi and amplitude 1
    return 2 * np.abs(x % (2 * math.pi) - math.pi) / math.pi - 1

def pre_ekg_wave(x):
    #returns the value of a EKG-like wave for one time x
    x = x % (2 * math.pi)
    if abs(x - math.pi) < 0.3:
        return 0.3 - abs(x - math.pi)
    if abs(x - math.pi + 1) < 0.3:
        return 0.03 * math.cos(abs(x - math.pi + 1) * math.pi / 0.6) ** 2
    if abs(x - math.pi - 1) < 0.3:
        return 0.03 * math.cos(abs(x - math.pi - 1) * math.pi / 0.6) ** 2
    else: return 0

def ekg_wave(x):
    #constructs and returns an EKG-like wave over time mesh x
    wave = np.zeros_like(x)
    for i in range(x.shape[0]):
        wave[i] = pre_ekg_wave(x[i])
    return (wave - 0.10799999999999998 / (2 * math.pi)) / np.sqrt(0.016548621676404782)

def make_omega_mesh(t_mesh, alpha):
    #constructs a logarithmic mesh for omega (frequency), spacing is 2^(1/20)
    #lowest frequency wave has 10*alpha radians spanned in the time mesh
    #highest frequency wave has 1 radian spanned over each increment in the time mesh
    ss = t_mesh[1] - t_mesh[0]
    tL = t_mesh[0]
    tU = t_mesh[-1]

    log_omegaL = round(math.log2(7.0 * alpha / (tU - tL)) * 20.0)
    log_omegaU = round(math.log2(1.0 / ss) * 20.0)

    omega_mesh = 2 ** (np.arange(log_omegaL, log_omegaU, dtype=np.float) / 20.0)
    return omega_mesh

def omega_compress_ratio(ss, omega):
    #In energy computations, we only compute at every compress_ratio number of time mesh points at each omega
    #If compress_ratio = 8 at some omega, we compute energy at every 8th time mesh point
    compress_ratio = max(1, 2 ** (round(math.log2(0.5 / (omega * ss)))))
    return compress_ratio

def compress_signal(signal, compress_ratio):
    sig_size = signal.shape[0]
    csig_size = int(compress_ratio * (sig_size // compress_ratio))

    csig = signal[:csig_size]
    csig = csig.reshape((-1, compress_ratio))
    csig = np.mean(csig, axis=1)

    return csig

def plot_E_cut(t_mesh, omega_mesh, E, omega_cut):
    #returns a heat-map plot of E
    N = t_mesh.shape[0]
    omega_N = omega_mesh.shape[0]
    aspect_ratio = 3.0 * np.log(omega_mesh[-1] / omega_mesh[0]) / (t_mesh[-1] - t_mesh[0])

    E2 = copy.deepcopy(E)
    max_E = np.max(E)
    for t_indx in range(N - 1, 2 * N - 1):
        omega_cut_indx_L = int(omega_cut[t_indx, 0])
        omega_cut_indx_H = int(omega_cut[t_indx, 1])
        if omega_cut_indx_L < omega_N:
            E2[omega_cut_indx_L, t_indx] = max_E
        if omega_cut_indx_H < omega_N:
            E2[omega_cut_indx_H, t_indx] = max_E

    plt.imshow(E2, cmap="gray_r", origin='lower', extent=(-1, 1, np.log10(omega_mesh[0]), np.log10(omega_mesh[-1])), aspect=aspect_ratio)
    plt.colorbar()
    plt.show()


def make_gabor_waves(t_mesh2, omega_mesh, alpha):
    #makes gabor waves for at each frequency in omega_mesh
    omega_N = omega_mesh.shape[0]
    gabor_waves = np.zeros((omega_N, 2, t_mesh2.shape[0]))
    for omega_indx in range(omega_N):
        omega = omega_mesh[omega_indx]

        gabor_waves[omega_indx, 0] = np.cos(omega * (t_mesh2)) * np.exp(-np.square(omega * (t_mesh2) / alpha))
        gabor_waves[omega_indx, 1] = np.sin(omega * (t_mesh2)) * np.exp(-np.square(omega * (t_mesh2) / alpha))
    return gabor_waves

def phase_modtosmooth(phase):
    #converts phase from [-pi, pi] to continuous
    phase = phase.reshape((-1))
    for t_indx in range(phase.shape[0] - 1):
        if np.abs(phase[t_indx + 1] - phase[t_indx]) > math.pi:
            phase[t_indx + 1:] -= ((phase[t_indx + 1] - phase[t_indx] + math.pi) // (2 * math.pi)) * (2 * math.pi)

    return phase

def phase_smoothtomod(phase):
    #converts phase from continuous to [-pi, pi]
    return (phase + math.pi) % (2 * math.pi) - math.pi

def max_squeeze_refine(t_mesh, omega_mesh, signal, t_indx, alpha):
    #Computes and returns the max squeezed frequency at one point (indexed by t_indx) in the t_mesh
    #Computes energies at each point within omega_mesh
    #In usage, omega_mesh is a localized mesh around an already identified approximate max squeezing (hence "refine")
    N = t_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]
    omega_N = omega_mesh.shape[0]
    t0 = t_mesh[t_indx]

    W_t = np.zeros((omega_N, 2))
    E_t = np.zeros((omega_N))

    for omega_indx in range(omega_N):
        omega = omega_mesh[omega_indx]
        bounds = int(5 * alpha / (omega * ss))

        int_low = max(0, t_indx - bounds)
        int_high = min(N, t_indx + bounds)

        t_mesh_i = t_mesh[int_low:int_high]

        sig_i = signal[int_low:int_high]
        cos_wave = np.cos(omega * (t_mesh_i - t0)) * np.exp(-np.square(omega * (t_mesh_i - t0) / alpha))
        sin_wave = np.sin(omega * (t_mesh_i - t0)) * np.exp(-np.square(omega * (t_mesh_i - t0) / alpha))

        W_t[omega_indx, 0] = np.dot(sin_wave, sig_i) * ss * omega
        W_t[omega_indx, 1] = np.dot(cos_wave, sig_i) * ss * omega

        E_t[omega_indx] = W_t[omega_indx, 0] ** 2 + W_t[omega_indx, 1] ** 2

    omega_max = np.argmax(E_t)
    omega_max = omega_mesh[omega_max]

    return E_t, W_t, omega_max

def loc_wave(t_mesh, wave_params, modes, t_indx):
    #Computes a local estimate for a mode
    #This is used to peel of identified mode segments (temp_modes in this implementation)
    if modes.shape[0] == 0:
        return 0
    N = t_mesh.shape[0]

    t0 = t_mesh[t_indx]
    loc_sig = np.zeros((N))

    for mode_indx in range(modes.shape[0]):
        omega = modes[mode_indx, t_indx, 0, 0]
        loc_theta = np.arctan2(-modes[mode_indx, t_indx, 1, 1], modes[mode_indx, t_indx, 1, 0])
        loc_amp = math.sqrt(modes[mode_indx, t_indx, 1, 1] ** 2 + modes[mode_indx, t_indx, 1, 0] ** 2)
        wave_params_i = wave_params.w
        if wave_params.w[0] == "unk":
            wave_params_i = ["custom", wave_params.t[mode_indx]]
        loc_sig += loc_amp * wave(wave_params_i, omega * (t_mesh - t0) + loc_theta)

    return loc_sig

def compute_E_border(t_mesh, omega_mesh, alpha):
    #Energy computation over full t_mesh and omega_mesh
    #Energy computations are not accurate near borders (of the time mesh), and are not computed

    N = t_mesh.shape[0]
    omega_N = omega_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]

    E_full = np.zeros((omega_N, N))


    for omega_indx in range(omega_N):
        omega = omega_mesh[omega_indx]
        cr = omega_compress_ratio(ss, omega)

        for t_indx in range(N // cr):
            t_indx *= cr
            bounds = int(5.0 * alpha / (omega * ss))

            if 0 <= t_indx <= N:
                E_full[omega_indx, t_indx:t_indx + cr] = 1

    return E_full

def compute_E_fast(t_mesh, omega_mesh, wave_params, sig_full, final_modes, temp_modes, E_thresh, alpha):
    #Computes E only at frequencies above previously identified full modes and temp modes
    #Stops computation 3 steps after a local max above E_thresh has been detected
    #TODO: Could be made faster by efficiently downsampling the signal at low frequencies
    N = t_mesh.shape[0]
    omega_N = omega_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]
    tL = t_mesh[0]
    tU = t_mesh[-1]
    t_mesh2 = np.arange(2 * tL, 2 * tU + ss, ss)

    E_full = np.zeros((omega_N, N))
    E_max_t = np.zeros((N))
    E_stop = np.zeros((N))
    min_freq = np.zeros((N))

    for mode_indx in range(final_modes.shape[0]):
        for t_i in range(N):
            min_freq[t_i] = max(min_freq[t_i], final_modes[mode_indx, t_i, 2])

    for mode_indx in range(temp_modes.shape[0]):
        for t_i in range(N):
            min_freq[t_i] = max(min_freq[t_i], temp_modes[mode_indx, t_i, 0, 0])

    min_freq *= 1.2 #The lowest frequency to compute E at any given time

    print("Energy computation progress:")

    for om_i in range(omega_N):
        omega = omega_mesh[om_i]
        cr = omega_compress_ratio(ss, omega)
        if om_i % (omega_N // 20) == 0: print(str(round((100 * om_i) / omega_N)) + "%", cr, time.localtime())


        bounds = int(5.0 * alpha / (omega * ss))
        sin_wave_full = np.sin(omega * (t_mesh2)) * np.exp(-np.square(omega * (t_mesh2) / alpha))
        cos_wave_full = np.cos(omega * (t_mesh2)) * np.exp(-np.square(omega * (t_mesh2) / alpha))

        for t_i in range(N // cr):
            t_i *= cr

            if 0 <= t_i <= N:
                if E_stop[t_i] < 3 and omega >= min_freq[t_i]:
                    int_low = max(0, t_i - bounds)
                    int_high = min(N, t_i + bounds)

                    int_low_full = int(int_low - t_i + N - 1)
                    int_high_full = int(int_high - t_i + N - 1)

                    sig_i = (sig_full - loc_wave(t_mesh, wave_params, temp_modes, t_i))[int_low:int_high]
                    sin_wave = sin_wave_full[int_low_full:int_high_full]
                    cos_wave = cos_wave_full[int_low_full:int_high_full]

                    E_full[om_i, t_i:t_i + cr] = (np.dot(sin_wave, sig_i) ** 2 + np.dot(cos_wave, sig_i) ** 2)
                    E_full[om_i, t_i:t_i + cr] *= (ss * omega) ** 2
                    E_full[om_i, t_i:t_i + cr] **= 0.5

                    E_max_t[t_i:t_i + cr] = np.maximum(E_full[om_i, t_i:t_i + cr], E_max_t[t_i:t_i + cr])

                    if E_thresh > 0:
                        if E_max_t[t_i] > E_thresh:
                            if E_full[om_i, t_i] < E_max_t[t_i]:
                                E_stop[t_i:t_i + cr] += 1
                            if E_full[om_i, t_i] == E_max_t[t_i]:
                                E_stop[t_i:t_i + cr] = 0

    return E_full

def micloc_KMD(t_mesh, signal, max_pow, t_indx, omega, alpha):
    #Computes a micro-local KMD of the signal on the t_indx-th point in the time mesh
    #Uses estimated frequency omega to generate the Gaussian Process
    N = t_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]

    t0 = t_mesh[t_indx]
    bounds = int(5 * alpha / (omega * ss))

    int_low = max(0, t_indx - bounds)
    int_high = min(N, t_indx + bounds)
    t_mesh_i = t_mesh[int_low:int_high]

    sig = signal[int_low:int_high] * np.exp(-np.square(omega * (t_mesh_i - t0) / alpha))
    chi_i = np.zeros((2 * max_pow + 2, int_high - int_low))

    for pow_i in range(max_pow + 1):
        sin_wave = np.sin(omega * (t_mesh_i - t0))
        cos_wave = np.cos(omega * (t_mesh_i - t0))

        chi_i[2 * pow_i] = cos_wave * np.exp(-np.square(omega * (t_mesh_i - t0) / alpha)) * (t0 - t_mesh_i) ** pow_i
        chi_i[2 * pow_i + 1] = sin_wave * np.exp(-np.square(omega * (t_mesh_i - t0) / alpha)) * (t0 - t_mesh_i) ** pow_i


    CS_mat = np.dot(chi_i, chi_i.T)
    W_cs = np.dot(chi_i, sig)
    [W_cs, *_] = LA.lstsq(CS_mat, W_cs, rcond=None)

    return W_cs.reshape((max_pow + 1, 2))

def micloc_KMD_final(t_mesh, signal, max_pow, t_indx, omega, theta, alpha):
    #Computes a micro-local KMD of the signal on the t_indx-th point in the time mesh
    #Uses estimated phase to construct the Gaussian Process
    N = t_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]

    t0 = t_mesh[t_indx]
    bounds = int(5 * alpha / (omega * ss))

    int_low = max(0, t_indx - bounds)
    int_high = min(N, t_indx + bounds)
    t_mesh_i = t_mesh[int_low:int_high]

    sig = signal[int_low:int_high] * np.exp(-np.square(omega * (t_mesh_i - t0) / alpha))
    chi_i = np.zeros((2 * max_pow + 2, int_high - int_low))

    for pow_i in range(max_pow + 1):
        sin_wave = np.sin(theta[int_low:int_high])
        cos_wave = np.cos(theta[int_low:int_high])

        chi_i[2 * pow_i] = cos_wave * np.exp(-np.square(omega * (t_mesh_i - t0) / alpha)) * (t0 - t_mesh_i) ** pow_i
        chi_i[2 * pow_i + 1] = sin_wave * np.exp(-np.square(omega * (t_mesh_i - t0) / alpha)) * (t0 - t_mesh_i) ** pow_i


    CS_mat = np.dot(chi_i, chi_i.T)
    W_cs = np.dot(chi_i, sig)
    [W_cs, *_] = LA.lstsq(CS_mat, W_cs, rcond=None)

    return W_cs.reshape((max_pow + 1, 2))


def get_raw_mode_fast(t_mesh, omega_mesh, wp, sig, final_modes, temp_modes, E_thresh, E_border, alpha, check_freq):
    #Returns a frequency and sin/cos (equiv to amp/phase) estimate
    #The frequency will be the lowest local max of E above previously identified final and temp modes at each time
    #temp modes are peeled here, final modes are not
    #Estimates sin/cos every check_freq rotations
    #Returns 0 at time t if no mode is detected at time t (no energies above E_thresh)

    N = t_mesh.shape[0]
    omega_N = omega_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]

    E = compute_E_fast(t_mesh, omega_mesh, wp, sig, final_modes, temp_modes, E_thresh, alpha)


    omega_max = np.zeros((N))

    #max_peak = np.zeros((3 * N - 2))
    t_indx = 0
    cond = True
    while cond and t_indx < N:

        peaks_x, _ = scisig.find_peaks(E[:, t_indx], height=E_thresh, distance=3)

        if peaks_x.shape[0] > 0:
            min_peak_x = int(np.min(peaks_x))

            omega_max[t_indx] = omega_mesh[min_peak_x]
            if np.sum(E_border[min_peak_x, :t_indx] ** 2) == 0:
                omega_max[:t_indx] = 0
            if np.sum(E_border[min_peak_x, t_indx + 1:]) == 0:
                omega_max[t_indx + 1:] = 0

                cond = False

        else:
            omega_max[t_indx] = -1

        t_indx += 1

    omega_mesh_fine = np.exp(0.002 * (np.arange(31) - 15))

    modes = np.zeros((N, 2, 2))
    rots = 0.0
    counter = 0
    for t_indx in range(N):
        if omega_max[t_indx] > 0:
            rots += omega_max[t_indx] * ss
            if int(rots / check_freq) != int((rots - omega_max[t_indx] * ss) / check_freq):
                sig_t = sig - loc_wave(t_mesh, wp, temp_modes, t_indx)

                omega_mesh_t = omega_max[t_indx] * omega_mesh_fine
                E_t, W_t, omega_max[t_indx] = max_squeeze_refine(t_mesh, omega_mesh_t, sig_t, t_indx, alpha)

                modes[counter, 0, 0] = omega_max[t_indx]
                modes[counter, 0, 1] = t_indx

                modes[counter, 1] = micloc_KMD(t_mesh, sig_t, 2, t_indx, omega_max[t_indx], alpha)[0]# / fund_coeff(wave_params)

                counter += 1

    modes = modes[:counter]

    return modes, E


def compute_loc_turb_energy(t_mesh, mode0, mode1):
    #computes the "energy" between an estimated freq/sin/cos at t0 and t1
    #high energy means the sin/cos is not consistent with freq or the freq/amplitude rapidly changes
    #high energy pairs of points are used to seperate the raw mode

    ss = t_mesh[1] - t_mesh[0]
    mode0 = mode0.reshape((2, 2))
    mode1 = mode1.reshape((2, 2))
    t0 = t_mesh[int(mode0[0, 1])]
    t1 = t_mesh[int(mode1[0, 1])]

    loc_turb_energy = 0


    phase0 = math.atan2(-mode0[1, 1], mode0[1, 0])
    phase1 = math.atan2(-mode1[1, 1], mode1[1, 0])
    #phase_est1 = ss * (mode0[0, 0] + mode1[0, 0]) / 2

    amp0 = math.sqrt(mode0[1, 1] ** 2 + mode0[1, 0] ** 2)
    amp1 = math.sqrt(mode1[1, 1] ** 2 + mode1[1, 0] ** 2)

    dphase_est = (t1 - t0) * (mode0[0, 0] + mode1[0, 0]) / 2
    dphase = (phase1 - phase0 + math.pi) % (2 * math.pi) - math.pi

    if amp0 > 0:
        loc_turb_energy = (30.0 * (amp0 - amp1) / amp0) ** 2
    if dphase != 0:
        loc_turb_energy += (30.0 * (dphase - dphase_est) / dphase) ** 2
    if mode0[0, 0] <= 0 or mode1[0, 0] <= 0:
        loc_turb_energy += 100
    else:
        loc_turb_energy += np.abs(np.log10(mode0[0, 0] / mode1[0, 0]) / np.log10(1.05))

    return loc_turb_energy

def mode_consolidate(t_mesh, omega_mesh, wave_params, sig, temp_modes, modes_all, alpha):
    #This returns a grouping of raw mode estimates (contained in modes_all)
    #Each grouping will be continuous and have consistent phase and frequency estimates
    #temp_modes peeled off here

    N = t_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]
    om_L = omega_mesh[0]
    om_H = omega_mesh[-1]

    modes_all = modes_all.reshape((-1, 2, 2))

    modes_bound = np.zeros((0, modes_all.shape[0], 2, 2))
    t_low = 0
    t_high = 0

    for t_i in range(modes_all.shape[0] - 1):
        if compute_loc_turb_energy(t_mesh, modes_all[t_i], modes_all[t_i + 1]) < 1:
            if om_L <= modes_all[t_i, 0, 0] <= om_H:
                t_high = t_i + 1
        else:
            if t_high - t_low > 5:
                omega = np.mean(modes_all[t_low + 1: t_high - 1, 0, 0])

                if modes_all[t_high, 0, 1] - modes_all[t_low + 1, 0, 1] > 1.0 * math.pi / (ss * omega):
                    modes_bound_i = np.zeros((1, modes_all.shape[0], 2, 2))
                    modes_bound_i[0, t_low + 1:t_high + 1] = modes_all[t_low + 1:t_high + 1]

                    modes_bound = np.concatenate((modes_bound, modes_bound_i), axis=0)

            t_low = copy.deepcopy(t_i)
            t_high = copy.deepcopy(t_i)

    if t_high - t_low > 5:
        omega = np.mean(modes_all[t_low + 1: t_high - 1, 0, 0])

        if modes_all[t_high, 0, 1] - modes_all[t_low + 1, 0, 1] > 1.0 * math.pi / (ss * omega):
            modes_bound_i = np.zeros((1, modes_all.shape[0], 2, 2))
            modes_bound_i[0, t_low + 1:t_high + 1] = modes_all[t_low + 1:t_high + 1]

            modes_bound = np.concatenate((modes_bound, modes_bound_i), axis=0)

    mode_frags = np.zeros((modes_bound.shape[0], N, 2, 2))

    for mode_indx in range(modes_bound.shape[0]):
        mode_t_min = np.min(np.nonzero(modes_bound[mode_indx, :, 0, 0])[0])
        mode_t_max = np.max(np.nonzero(modes_bound[mode_indx, :, 0, 0])[0])

        t_min = int(modes_bound[mode_indx, mode_t_min, 0, 1])
        t_max = int(modes_bound[mode_indx, mode_t_max, 0, 1])

        t_mesh_i = np.zeros((mode_t_max - mode_t_min + 1))

        for t_i in range(mode_t_min, mode_t_max + 1):
            t_mesh_i[t_i - mode_t_min] = t_mesh[int(modes_bound[mode_indx, t_i, 0, 1])]

        sin_frag = -modes_bound[mode_indx, mode_t_min:mode_t_max + 1, 1, 1]
        cos_frag = modes_bound[mode_indx, mode_t_min:mode_t_max + 1, 1, 0]

        amp_frag0 = np.sqrt(cos_frag ** 2 + sin_frag ** 2)
        phase_frag0 = np.arctan2(sin_frag, cos_frag)
        phase_frag0 = phase_modtosmooth(phase_frag0)

        freq_frag_spl = CubicSpline(t_mesh_i, modes_bound[mode_indx, mode_t_min:mode_t_max + 1, 0, 0])
        amp_frag_spl = CubicSpline(t_mesh_i, amp_frag0)
        phase_frag_spl = CubicSpline(t_mesh_i, phase_frag0)

        freq_frag = freq_frag_spl(t_mesh[t_min:t_max + 1])
        amp_frag = amp_frag_spl(t_mesh[t_min:t_max + 1])
        phase_frag = phase_frag_spl(t_mesh[t_min:t_max + 1])

        mode_frags[mode_indx, t_min:t_max + 1, 0, 0] = freq_frag
        mode_frags[mode_indx, t_min:t_max + 1, 0, 1] = np.arange(t_min, t_max + 1)
        mode_frags[mode_indx, t_min:t_max + 1, 1, 0] = amp_frag * np.cos(phase_frag)
        mode_frags[mode_indx, t_min:t_max + 1, 1, 1] = -amp_frag * np.sin(phase_frag)

    modes_bound = copy.deepcopy(mode_frags)
    if modes_bound.shape[0] > 0:
        mode_t_min = np.min(np.nonzero(modes_bound[0, :, 0, 0])[0])
        mode_t_max = np.max(np.nonzero(modes_bound[-1, :, 0, 0])[0])

        omega_low = modes_bound[0, mode_t_min, 0, 0]
        omega_high = modes_bound[-1, mode_t_max, 0, 0]

        mode_keep = np.zeros((modes_bound.shape[0]))

        for mode_indx in range(modes_bound.shape[0]):
            mode_t_min = np.min(np.nonzero(modes_bound[mode_indx, :, 0, 0])[0])
            mode_t_max = np.max(np.nonzero(modes_bound[mode_indx, :, 0, 0])[0])

            smooth = True
            mode0 = modes_bound[mode_indx, mode_t_min]

            amp_max = np.max(np.sqrt(modes_bound[mode_indx, :, 1, 0] ** 2 + modes_bound[mode_indx, :, 1, 1] ** 2))
            #print("test1:", mode_t_min)
            while smooth and mode_t_min > 0:
                mode1 = border_KMD3(t_mesh, wave_params, sig, mode0, temp_modes, mode_t_min - 1, 1, alpha)
                smooth = False
                # TODO: Make this vanishing condition better
                if mode1[0, 0] > 0:
                    if np.sqrt(mode1[1, 0] ** 2 + mode1[1, 1] ** 2) > 0.01 * amp_max:
                        modes_bound[mode_indx, mode_t_min - 1] = mode1
                        mode0 = copy.deepcopy(mode1)
                        mode_t_min -= 1
                        smooth = True

            smooth = True
            mode0 = modes_bound[mode_indx, mode_t_max]

            while smooth and mode_t_max < N - 1:
                mode1 = border_KMD3(t_mesh, wave_params, sig, mode0, temp_modes, mode_t_max + 1, 1, alpha)
                smooth = False

                if mode1[0, 0] > 0:
                    if np.sqrt(mode1[1, 0] ** 2 + mode1[1, 1] ** 2) > 0.01 * amp_max:
                        modes_bound[mode_indx, mode_t_max + 1] = mode1
                        mode0 = copy.deepcopy(mode1)
                        mode_t_max += 1
                        smooth = True
            omega_avg = np.mean(modes_bound[mode_indx, mode_t_min:mode_t_max, 0, 0])
            if mode_t_max - mode_t_min > 20.0 * math.pi / (ss * omega_avg):
                mode_keep[mode_indx] = 1

    num_modes = np.count_nonzero(mode_keep)
    modes_c = np.zeros((num_modes, N, 2, 2))
    counter = 0
    for mode_indx in range(modes_bound.shape[0]):
        if mode_keep[mode_indx] != 0:
            modes_c[counter] = modes_bound[mode_indx]
            counter += 1
    return modes_c

def border_KMD3(t_mesh, wave_params, signal, mode0, temp_modes, t_indx1, thresh, alpha):
    #Inputs the estimate of a mode at t_indx0 (mode0), which should be at an edge of a submode
    #Determines whether the mode can be continuously extended to t_indx1 (a neighboring point in the time mesh)
    #If it can be extended, it returns a mode estimate at t_indx1
    #If not, it returns a 0 mode

    signal = signal - loc_wave(t_mesh, wave_params, temp_modes, t_indx1)

    mode0 = mode0.reshape((1, 2, 2))

    omega_mesh_fine = np.exp(0.002 * (np.arange(31) - 15)) * mode0[0, 0, 0]

    E_t, W_t, omega_max = max_squeeze_refine(t_mesh, omega_mesh_fine, signal, t_indx1, alpha)

    mode1 = np.zeros((1, 2, 2))
    mode1[0, 0, 0] = omega_max
    mode1[0, 0, 1] = t_indx1

    mode1[0, 1:, :] = micloc_KMD(t_mesh, signal, 2, t_indx1, omega_max, alpha)[0]

    turb_energy = compute_loc_turb_energy(t_mesh, mode0, mode1)

    if turb_energy < thresh:
        return mode1.reshape((2, 2))
    else:
        return np.zeros((2, 2))

def convert_to_final_mode(t_mesh, wave_params, raw_mode, signal, temp_modes, thresh, alpha):
    #converts raw_mode, which is in the form sin/cos, to amp/phase
    #extends the mode if possible
    #returns a final mode
    N = t_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]

    t_L = np.min(np.nonzero(raw_mode[0, :, 0, 0])[0])
    t_H = np.max(np.nonzero(raw_mode[0, :, 0, 0])[0])

    raw_mode = raw_mode.reshape((1, -1, 2, 2))
    fmode = np.zeros((1, raw_mode.shape[1], 4))

    fmode[0, :, 0] = np.sqrt(raw_mode[0, :, 1, 0] ** 2 + raw_mode[0, :, 1, 1] ** 2)
    fmode[0, :, 1] = phase_modtosmooth(np.arctan2(-raw_mode[0, :, 1, 1], raw_mode[0, :, 1, 0]))
    fmode[0, :, 2] = raw_mode[0, :, 0, 0]


    smooth = True
    mode0 = raw_mode[0, t_L]

    amp_max = np.max(fmode[0, :, 0])

    while smooth and t_L > 0:
        mode1 = border_KMD3(t_mesh, wave_params, signal, mode0, temp_modes, t_L - 1, thresh, alpha)
        smooth = False
        #TODO: Make this vanishing condition better
        if mode1[0, 0] > 0:
            if np.sqrt(mode1[1, 0] ** 2 + mode1[1, 1] ** 2) > 0.01 * amp_max:
                fmode[0, t_L - 1, 0] = np.sqrt(mode1[1, 0] ** 2 + mode1[1, 1] ** 2)
                fmode[0, t_L - 1, 1] = np.arctan2(-mode1[1, 1], mode1[1, 0])
                fmode[0, t_L - 1, 2] = mode1[0, 0]
                raw_mode[0, t_L - 1] = mode1
                mode0 = copy.deepcopy(mode1)
                t_L -= 1
                smooth = True


    smooth = True
    mode0 = raw_mode[0, t_H]

    while smooth and t_H < N - 1:
        mode1 = border_KMD3(t_mesh, wave_params, signal, mode0, temp_modes, t_H + 1, thresh, alpha)
        smooth = False

        if mode1[0, 0] > 0:
            if np.sqrt(mode1[1, 0] ** 2 + mode1[1, 1] ** 2) > 0.01 * amp_max:
                fmode[0, t_H + 1, 0] = np.sqrt(mode1[1, 0] ** 2 + mode1[1, 1] ** 2)
                fmode[0, t_H + 1, 1] = np.arctan2(-mode1[1, 1], mode1[1, 0])
                fmode[0, t_H + 1, 2] = mode1[0, 0]
                raw_mode[0, t_H + 1] = mode1
                mode0 = copy.deepcopy(mode1)
                t_H += 1
                smooth = True

    t_L = np.min(np.nonzero(raw_mode[0, :, 0, 0])[0])
    t_H = np.max(np.nonzero(raw_mode[0, :, 0, 0])[0])

    if t_L != 0:
        omega_low = fmode[0, t_L, 2]
        wind_low = int(10 * math.pi / (omega_low * ss))
        amp_low = np.min(fmode[0, t_L:t_L + wind_low, 0])
        amp_max = np.max(fmode[0, t_L:t_H, 0])

        if amp_low / amp_max < 0.5 or t_L > wind_low:
            fit_low = np.polyfit(t_mesh[t_L:t_L + wind_low], fmode[0, t_L:t_L + wind_low, 0], 1)
            if fit_low[0] > 0:
                t_L2 = max(0, t_L - int(2 * fmode[0, t_L, 0] / (ss * fit_low[0])))
                t_min = t_mesh[t_L] - 2 * fmode[0, t_L, 0] / fit_low[0]

                fmode[0, t_L2:t_L, 0] = fmode[0, t_L, 0] * ((t_mesh[t_L2:t_L] - t_min) / (t_mesh[t_L] - t_min)) ** 2
                fmode[0, :t_L, 1] = fmode[0, t_L, 2] * (t_mesh[:t_L] - t_mesh[t_L]) + fmode[0, t_L, 1]
                fmode[0, :t_L, 2] = fmode[0, t_L, 2]
                fmode[0, 0, 3] = t_L2

            else:
                t_L2 = max(0, t_L - wind_low)
                t_min = t_mesh[t_L] - wind_low * ss

                fmode[0, t_L2:t_L, 0] = fmode[0, t_L, 0] * ((t_mesh[t_L2:t_L] - t_min) / (t_mesh[t_L] - t_min)) ** 2
                fmode[0, :t_L, 1] = fmode[0, t_L, 2] * (t_mesh[:t_L] - t_mesh[t_L]) + fmode[0, t_L, 1]
                fmode[0, :t_L, 2] = fmode[0, t_L, 2]
                fmode[0, 0, 3] = t_L2

        else:
            fit_low0 = np.polyfit(t_mesh[t_L:t_L + wind_low], fmode[0, t_L:t_L + wind_low, 0], 1)
            fit_low2 = np.polyfit(t_mesh[t_L:t_L + wind_low], fmode[0, t_L:t_L + wind_low, 2], 1)

            fmode[0, :t_L, 0] = fit_low0[1] + fit_low0[0] * t_mesh[:t_L]
            fmode[0, :t_L, 2] = fit_low2[1] + fit_low2[0] * t_mesh[:t_L]
            for t_indx in range(t_L):
                t_indx = t_L - 1 - t_indx
                fmode[0, t_indx, 1] = fmode[0, t_indx + 1, 1] - fmode[0, t_indx + 1, 2] * ss
            fmode[0, 0, 3] = 0

    else:
        fmode[0, 0, 3] = 0

    if t_H != N - 1:
        omega_high = fmode[0, t_H, 2]
        wind_high = int(10 * math.pi / (omega_high * ss))
        amp_high = np.min(fmode[0, t_H - wind_high:t_H, 0])
        amp_max = np.max(fmode[0, t_L:t_H, 0])

        if amp_high / amp_max < 0.5 and t_H < N - wind_high:
            fit_high = np.polyfit(t_mesh[t_H - wind_high:t_H], fmode[0, t_H - wind_high:t_H, 0], 1)
            if fit_high[0] > 0:
                t_H2 = min(N, t_H + int(2 * fmode[0, t_H, 0] / (ss * fit_high[0])))
                t_max = t_mesh[t_H] - 2 * fmode[0, t_H, 0] / fit_high[0]

                fmode[0, t_H:t_H2, 0] = fmode[0, t_H, 0] * ((t_mesh[t_H:t_H2] - t_max) / (t_mesh[t_H] - t_max)) ** 2
                fmode[0, t_H:, 1] = fmode[0, t_H, 2] * (t_mesh[t_H:] - t_mesh[t_H]) + fmode[0, t_H, 1]
                fmode[0, t_H:, 2] = fmode[0, t_H, 2]
                fmode[0, 1, 3] = t_H2

            else:
                t_H2 = max(0, t_H + wind_high)
                t_max = t_mesh[t_H] + wind_high * ss

                fmode[0, t_H:t_H2, 0] = fmode[0, t_H, 0] * ((t_mesh[t_H:t_H2] - t_max) / (t_mesh[t_H] - t_max)) ** 2
                fmode[0, t_H:, 1] = fmode[0, t_H, 2] * (t_mesh[t_H:] - t_mesh[t_H]) + fmode[0, t_H, 1]
                fmode[0, t_H:, 2] = fmode[0, t_H, 2]
                fmode[0, 1, 3] = t_H2

        else:
            t_H += 1
            fit_high0 = np.polyfit(t_mesh[t_H - wind_high:t_H], fmode[0, t_H - wind_high:t_H, 0], 1)
            fit_high2 = np.polyfit(t_mesh[t_H - wind_high:t_H], fmode[0, t_H - wind_high:t_H, 2], 1)

            fmode[0, t_H:, 0] = fit_high0[1] + fit_high0[0] * t_mesh[t_H:]
            fmode[0, t_H:, 2] = fit_high2[1] + fit_high2[0] * t_mesh[t_H:]
            for t_indx in range(t_H, N):
                fmode[0, t_indx, 1] = fmode[0, t_indx - 1, 1] + fmode[0, t_indx, 2] * ss
            fmode[0, 1, 3] = N
    else:
        fmode[0, 1, 3] = N

    fmode[0, :, 0] = np.maximum(0, fmode[0, :, 0])
    fmode[0, :, 1] = phase_modtosmooth(fmode[0, :, 1])

    #print(final_mode[0, 0:2, 3])
    return fmode

def mode_connect3(t_mesh, mode0, mode1, rot):
    #connects two mode fragments (fills the gaps in time between them)
    #amplitudes and frequencies are linearly connected in the gap
    #phase in the gap is initially estimated integrating interpolated frequency, then to match both ends, a tweak is added
    #rot adds additional rotations in the gap with tweak, default is 0
    N = t_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]

    mode0 = mode0.reshape((1, N, 2, 2))
    mode1 = mode1.reshape((1, N, 2, 2))

    conn_mode = np.zeros((1, N, 2, 2))

    mode0_t_low = np.min(np.nonzero(mode0[0, :, 0, 0])[0])
    mode1_t_low = np.min(np.nonzero(mode1[0, :, 0, 0])[0])
    mode0_t_high = np.max(np.nonzero(mode0[0, :, 0, 0])[0])
    mode1_t_high = np.max(np.nonzero(mode1[0, :, 0, 0])[0])

    if mode0_t_low < mode1_t_low:
        cut_low = mode0_t_high + 1
        cut_high = mode1_t_low

        conn_mode[:, :cut_low] = mode0[:, :cut_low]
        conn_mode[:, cut_high:] = mode1[:, cut_high:]

    else:
        cut_low = mode1_t_high + 1
        cut_high = mode0_t_low

        conn_mode[:, :cut_low] = mode1[:, :cut_low]
        conn_mode[:, cut_high:] = mode0[:, cut_high:]

    pi = math.pi

    amp_low = np.sqrt(conn_mode[:, cut_low - 1, 1, 0] ** 2 + conn_mode[:, cut_low - 1, 1, 1] ** 2)
    amp_high = np.sqrt(conn_mode[:, cut_high, 1, 0] ** 2 + conn_mode[:, cut_high, 1, 1] ** 2)

    phase_low = (np.arctan2(-conn_mode[:, cut_low - 1, 1, 1], conn_mode[:, cut_low - 1, 1, 0]) + pi) % (2 * pi) - pi
    phase_high = (np.arctan2(-conn_mode[:, cut_high, 1, 1], conn_mode[:, cut_high, 1, 0]) + pi) % (2 * pi) - pi

    freq_low = conn_mode[:, cut_low - 1, 0, 0]
    freq_high = conn_mode[:, cut_high, 0, 0]

    dphase = phase_high - phase_low
    for t_indx in range(cut_low, cut_high + 1):
        dphase -= (freq_low + ((t_indx - cut_low + 1) / (cut_high - cut_low + 1)) * (freq_high - freq_low)) * ss

    phase_t = copy.deepcopy(phase_low)

    dphase = (dphase + math.pi) % (2 * math.pi) - math.pi + 2 * math.pi * rot
    t_mid = (t_mesh[cut_low - 1] + t_mesh[cut_high]) / 2.0

    if cut_high - cut_low > 2:
        tweak_ratio = 0

        for t_indx in range(cut_low, cut_high):
            tweak_ratio += ss * ((t_mesh[cut_high] - t_mid) ** 2 - (t_mesh[t_indx] - t_mid) ** 2)

        tweak_freq = dphase / tweak_ratio
    else: tweak_freq = 0

    for t_indx in range(cut_low, cut_high):
        conn_mode[:, t_indx, 0, 1] = conn_mode[:, t_indx - 1, 0, 1] + 1
        conn_mode[:, t_indx, 0, 0] = freq_low + ((t_indx - cut_low + 1) / (cut_high - cut_low + 1)) * (
        freq_high - freq_low)
        conn_mode[:, t_indx, 0, 0] += tweak_freq * ((t_mesh[cut_high] - t_mid) ** 2 - (t_mesh[t_indx] - t_mid) ** 2)

        amp_t = amp_low + ((t_indx - cut_low + 1) / (cut_high - cut_low + 1)) * (amp_high - amp_low)
        phase_t = (phase_t + ss * conn_mode[:, t_indx, 0, 0] + math.pi) % (2 * math.pi) - math.pi

        conn_mode[:, t_indx, 1, 0] = amp_t * np.cos(phase_t)
        conn_mode[:, t_indx, 1, 1] = -amp_t * np.sin(phase_t)

    return conn_mode

def final_mode_group3(t_mesh, wave_params, signal, modes, temp_modes, rot, alpha):
    #inputs a mode segments to be combined into a final mode
    #first sorts the modes in time, then connects their gaps (with mode_connect3)
    #then converts to amp/phase and extends the mode if possible
    #returns a final mode

    N = t_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]

    num_modes = modes.shape[0]
    mode = np.zeros((1, N, 2, 2))
    fmode = np.zeros((1, N, 4))

    if num_modes == 1:
        mode_freq = modes[0, :, 0, 0]
        mode_indx_low = np.min(np.nonzero(mode_freq)[0])
        mode_indx_high = np.max(np.nonzero(mode_freq)[0])

        omega_low = modes[0, mode_indx_low, 0, 0]
        omega_high = modes[0, mode_indx_high, 0, 0]

        fmode[0] = convert_to_final_mode(t_mesh, wave_params, modes, signal, temp_modes, 1, alpha)



    else:
        num_modes_t = np.zeros((N))
        for mode_indx in range(modes.shape[0]):
            for t_indx in range(N):
                if modes[mode_indx, t_indx, 0, 0] != 0:
                    num_modes_t[t_indx] += 1

        if np.max(num_modes_t) == 1:
            mode_indx_low = np.zeros((modes.shape[0]))
            mode_indx_high = np.zeros((modes.shape[0]))

            for mode_indx in range(modes.shape[0]):
                mode_indx_low[mode_indx] = np.min(np.nonzero(modes[mode_indx, :, 0, 0])[0])
                mode_indx_high[mode_indx] = np.min(np.nonzero(modes[mode_indx, :, 0, 0])[0])

            mode_indx_sorted = np.argsort(mode_indx_low)
            mode[0] = modes[mode_indx_sorted[-1]]
            for mode_indx in range(num_modes - 1):
                next_mode_indx = mode_indx_sorted[-mode_indx - 2]
                mode[0] = mode_connect3(t_mesh, mode[0], modes[next_mode_indx], rot[mode_indx])
            fmode[0] = convert_to_final_mode(t_mesh, wave_params, mode, signal, temp_modes, 1, alpha)

        else:
            for t_indx in range(N):
                if num_modes_t[t_indx] > 1:
                    modes[:, t_indx] *= 0

            num_modes_t = np.zeros((N))
            for mode_indx in range(modes.shape[0]):
                for t_indx in range(N):
                    if modes[mode_indx, t_indx, 0, 0] != 0:
                        num_modes_t[t_indx] += 1

            mode_indx_low = np.zeros((modes.shape[0]))
            mode_indx_high = np.zeros((modes.shape[0]))

            for mode_indx in range(modes.shape[0]):
                mode_indx_low[mode_indx] = np.min(np.nonzero(modes[mode_indx, :, 0, 0])[0])
                mode_indx_high[mode_indx] = np.max(np.nonzero(modes[mode_indx, :, 0, 0])[0])

            mode_indx_sorted = np.argsort(mode_indx_low)
            mode[0] = modes[mode_indx_sorted[-1]]
            for mode_indx in range(num_modes - 1):
                mode[0] = mode_connect3(t_mesh, mode[0], mode_indx_sorted[-mode_indx - 2], rot[mode_indx])

            fmode[0] = convert_to_final_mode(t_mesh, wave_params, mode, signal, temp_modes, 1, alpha)

    return fmode

def represents_int(s):
    #returns True if s is a string representing an integer
    #returns False otherwise
    #courtesy of Triptych on Stackexchange
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_input_modedivide(t_mesh, omega_mesh, modes_freq, E, prompt):
    #controls the user input on which modes to group, designate as fragments, or trash
    N = t_mesh.shape[0]
    print(prompt)
    for mode_indx in range(modes_freq.shape[0]):
        t_L = np.min(np.nonzero(modes_freq[mode_indx])[0])
        t_H = np.max(np.nonzero(modes_freq[mode_indx])[0])

        plt.plot(t_mesh[t_L:t_H + 1], modes_freq[mode_indx, t_L:t_H + 1], label="Mode segment " + str(mode_indx))
    plt.legend()
    plt.show()

    lom_L = np.log10(omega_mesh[0])
    lom_H = np.log10(omega_mesh[-1])

    aspect_ratio = 2.2 * np.log(omega_mesh[-1] / omega_mesh[0]) / (t_mesh[-1] - t_mesh[0])

    plt.imshow(E, cmap="gray_r", origin='lower', extent=(-1, 1, lom_L, lom_H), aspect=aspect_ratio, vmin=0, vmax=85)
    plt.show()

    user_input = ""
    mode_divide = []
    remaining_modes = np.arange(modes_freq.shape[0])
    mode_i = np.zeros((0), dtype=np.int)

    while user_input != "Done":
        user_input = input("Input mode segments to add to mode " + str(len(mode_divide)) +
                           " (if done with mode: Next, if done with all modes: Done): ")
        if represents_int(user_input):
            if int(user_input) in remaining_modes:
                mode_i = np.concatenate((mode_i, [int(user_input)]), axis=0)
                #print(mode_i)
                remaining_modes = np.delete(remaining_modes, np.where(remaining_modes == int(user_input))[0][0], 0)
            else: print("Invalid mode selected, please try again")
        elif user_input == "All":
            mode_i = remaining_modes
            remaining_modes = np.zeros((0), dtype=np.int)
        elif user_input == "Next":
            if mode_i.shape[0] > 0:
                mode_divide.append(mode_i)
            mode_i = np.zeros((0), dtype=np.int)
        elif user_input != "Done": print("Invalid input, please try again")

    if mode_i.shape[0] > 0: mode_divide.append(mode_i)

    temp_modes = np.zeros((0), dtype=np.int)

    if remaining_modes.shape[0] > 0:
        print("The following modes remain, select which ones to keep:")
        for mode_indx in remaining_modes:
            t_L = np.min(np.nonzero(modes_freq[mode_indx])[0])
            t_H = np.max(np.nonzero(modes_freq[mode_indx])[0])

            plt.plot(t_mesh[t_L:t_H + 1], modes_freq[mode_indx, t_L:t_H + 1],
                     label="Mode " + str(mode_indx))
        plt.legend()
        plt.show()

        user_input = ""

        while user_input != "Done":
            user_input = input("Input mode segments to keep for next iteration " + str(
                len(mode_divide)) + " (if done: Done): ")
            if represents_int(user_input):
                if int(user_input) in remaining_modes:
                    # print(mode_i)
                    remaining_modes = np.delete(remaining_modes, np.where(remaining_modes == int(user_input))[0][0], 0)
                    temp_modes = np.concatenate((temp_modes, [int(user_input)]), axis=0)
                else:
                    print("Invalid mode selected, please try again")
            elif user_input == "All":
                temp_modes = remaining_modes
                remaining_modes = np.zeros((0), dtype=np.int)
            elif user_input != "Done":
                print("Invalid input, please try again")

    return mode_divide, temp_modes

def compute_freq(t_mesh, micloc_wave):
    #estimates frequency of a mode in sin/cos form
    ss = t_mesh[1] - t_mesh[0]

    phase = np.arctan2(-micloc_wave[:, 1], micloc_wave[:, 0])
    phase = phase_modtosmooth(phase)

    freq = np.zeros_like(t_mesh)
    freq[:-1] = (phase[1:] - phase[:-1]) / ss
    freq[-1] = freq[-2]

    return freq



def mode_clip(t_mesh, orig_freq, comp_freq, micloc_wave):
    #clips the mode near edges of support if the phase and frequency of the modes do not agree
    #They are very prone to disagree due to errors either due to being near the mesh border or a vanishing of the mode

    ss = t_mesh[1] - t_mesh[0]

    low_stop = np.min(np.nonzero(micloc_wave[:, 0] ** 2 + micloc_wave[:, 1] ** 2)[0])
    high_stop = np.max(np.nonzero(micloc_wave[:, 0] ** 2 + micloc_wave[:, 1] ** 2)[0]) + 1

    new_low_stop = 0
    new_high_stop = 0

    omega_low = orig_freq[low_stop]
    omega_high = orig_freq[high_stop - 1]

    wind_low = int(12 * math.pi / (omega_low * ss))
    wind_high = int(12 * math.pi / (omega_high * ss))

    micloc_freq = compute_freq(t_mesh, micloc_wave)

    freq_dev1 = micloc_freq / orig_freq
    freq_dev2 = (micloc_freq + comp_freq) / orig_freq

    t_indx = low_stop + wind_low
    #print(t_indx)
    cond = True

    while cond and t_indx > low_stop:
        if np.abs(freq_dev1[t_indx]) > 0.01 or not 1 / 1.1 < freq_dev2[t_indx] < 1.1:
            cond = False
            micloc_wave[:t_indx] = 0
            new_low_stop = copy.deepcopy(t_indx)
        t_indx -= 1
        if t_indx == low_stop:
            new_low_stop = copy.deepcopy(t_indx)

    t_indx = high_stop - wind_high
    #print(t_indx)
    cond = True

    while cond and t_indx < high_stop - 1:
        if np.abs(freq_dev1[t_indx]) > 0.01 or not 1 / 1.1 < freq_dev2[t_indx] < 1.1:
            cond = False
            micloc_wave[t_indx + 1:] = 0
            new_high_stop = copy.deepcopy(t_indx) + 1
        t_indx += 1
        if t_indx == high_stop - 1:
            new_high_stop = copy.deepcopy(t_indx) + 1
    #print(new_low_stop, new_high_stop)
    return micloc_wave, new_low_stop, new_high_stop


def mama_ker_eig(Chi, sigm):
    #efficiently computes the eigenbasis of a mama kernel (\sigma^2 I) + \sum \chi_i^T \chi_i
    basis_chi = scipy.linalg.orth(Chi)

    quiet_ker = np.zeros((basis_chi.shape[1], basis_chi.shape[1]))
    #mama ker expressed in chi subspace except for noisy component

    for i in range(Chi.shape[1]):
        chi_i = np.dot(basis_chi.T, Chi[:, i].reshape((-1, 1)))
        quiet_ker += np.dot(chi_i.reshape((-1, 1)), chi_i.reshape((1, -1)))

    w, v = LA.eigh(quiet_ker)

    quiet_ker_v = np.dot(basis_chi, v.T)
    quiet_ker_w = 1.0 / (sigm + w)


    return quiet_ker_w, quiet_ker_v

def learn_waveform0(t_mesh, sig, fmodes, final_mode_i, wave_params, num_overtones, alpha):
    #learns the waveform of final_mode_i, using fmodes to peel modes from sig
    #num_overtones: number of Fourier coefficients to estimate in the base waveform
    N = t_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]
    sig_iter = copy.deepcopy(sig)
    num_iter = fmodes.shape[0]
    final_mode_i = final_mode_i.reshape((1, N, 4))
    low_stop = int(final_mode_i[0, 0, 3])
    high_stop = int(final_mode_i[0, 1, 3])

    if num_iter >= 1:
        for iter_indx in range(num_iter):
            amp_indx = fmodes[iter_indx, :, 0]
            tau_indx = fmodes[iter_indx, :, 1]

            wave_params_i = ["custom", wave_params.f[iter_indx]]

            sig_iter -= amp_indx * wave(wave_params_i, tau_indx)

    theta1e = final_mode_i[0, :, 1]
    omega1e = final_mode_i[0, :, 2]
    overtones = np.zeros((N, 2 * num_overtones - 1))

    rots = np.random.uniform(0, 1)
    counter = 0
    check_freq = 0.2

    for t_indx in range(low_stop, high_stop):
        if t_indx % 1000 == 0: print(t_indx, time.localtime())
        omega = omega1e[t_indx]
        rots += omega * ss
        if int(rots / check_freq) != int((rots - omega * ss) / check_freq):
            t0 = t_mesh[t_indx]
            omega = omega1e[t_indx]
            bounds = int(5 * alpha / (omega * ss))

            int_low = max(0, t_indx - bounds)
            int_high = min(N, t_indx + bounds)

            t_mesh_i = t_mesh[int_low:int_high]
            cos_wave = np.zeros((num_overtones, int_high - int_low))
            sin_wave = np.zeros((num_overtones, int_high - int_low))
            for ov_indx in range(num_overtones):
                cos_wave[ov_indx] = np.cos((ov_indx + 1.0) * theta1e[int_low:int_high])
                sin_wave[ov_indx] = np.sin((ov_indx + 1.0) * theta1e[int_low:int_high])

            chi = np.zeros((2 * num_overtones - 1, int_high - int_low))

            for ov_indx in range(num_overtones):
                chi[ov_indx] = cos_wave[ov_indx] * np.exp(-np.square(omega * (t_mesh_i - t0) / alpha))

            for ov_indx in range(num_overtones - 1):
                chi[ov_indx + num_overtones] = sin_wave[ov_indx + 1] * np.exp(
                    -np.square(omega * (t_mesh_i - t0) / alpha))

            chiT = chi.T

            quiet_ker_w, quiet_ker_v = mama_ker_eig(chiT, 0.0001)

            invQsig = np.dot(quiet_ker_v.T, sig_iter[int_low:int_high])
            invQsig = invQsig.reshape((-1)) * quiet_ker_w.reshape((-1))
            invQsig = np.dot(quiet_ker_v, invQsig.reshape((-1, 1)))

            for ov_indx in range(2 * num_overtones - 1):
                overtones[counter, ov_indx] = np.dot(chi[ov_indx].reshape((1, -1)), invQsig.reshape((-1, 1)))

            counter += 1

    overtones = overtones[:counter]
    return overtones

def learn_waveform(t_mesh, sig, fmodes, temp_mode_i, wave_params, num_overtones, alpha):
    #Learn the waveform of a temp_mode_i (using learn_waveform0)
    N = t_mesh.shape[0]
    overtones = np.zeros((0, 2 * num_overtones - 1))
    for mode_indx in range(temp_mode_i.shape[0]):
        final_mode_i = np.zeros((1, N, 4))
        low_stop = np.min(np.nonzero(temp_mode_i[mode_indx, :, 0, 0])[0])
        high_stop = np.max(np.nonzero(temp_mode_i[mode_indx, :, 0, 0])[0]) + 1

        cos_mode = temp_mode_i[mode_indx, low_stop:high_stop, 1, 0]
        sin_mode = -temp_mode_i[mode_indx, low_stop:high_stop, 1, 1]

        final_mode_i[0, low_stop:high_stop, 0] = np.sqrt(cos_mode ** 2 + sin_mode ** 2)
        final_mode_i[0, low_stop:high_stop, 1] = np.arctan2(-sin_mode, cos_mode)
        final_mode_i[0, low_stop:high_stop, 1] = phase_modtosmooth(final_mode_i[0, low_stop:high_stop, 1])
        final_mode_i[0, low_stop:high_stop, 2] = temp_mode_i[mode_indx, low_stop:high_stop, 0, 0]
        final_mode_i[0, 0, 3] = low_stop
        final_mode_i[0, 1, 3] = high_stop

        overtones_i = learn_waveform0(t_mesh, sig, fmodes, final_mode_i, wave_params, num_overtones, alpha)
        overtones = np.concatenate((overtones, overtones_i), axis=0)

    a_i = np.ones((num_overtones, 2))
    a_i[0, 1] = 0
    for ov_indx in range(1, 2 * num_overtones - 1):
        overtone_i = overtones[:, ov_indx] / overtones[:, 0]
        bin_num = int((np.max(overtone_i) - np.min(overtone_i)) / 0.002) + 1
        hist, edges = np.histogram(overtone_i, bins=bin_num)

        hist_max = np.argmax(hist)

        if hist[hist_max] / np.sum(hist) > 0.05:
            overtone_i = overtone_i[overtone_i <= edges[hist_max + 1]]
            overtone_i = overtone_i[overtone_i >= edges[hist_max]]
            if ov_indx < num_overtones:
                a_i[ov_indx, 0] = np.mean(overtone_i)
            else:
                a_i[ov_indx - num_overtones + 1, 1] = np.mean(overtone_i)

        else:
            if ov_indx < num_overtones:
                a_i[ov_indx, 0] = 0
            else:
                a_i[ov_indx - num_overtones + 1, 1] = 0

    print(a_i)
    return a_i

def learn_waveform_final(t_mesh, sig_full, fmodes, mode_inter, wp, alpha):
    #refines the estimates of the waveforms of all modes in fmodes
    N = t_mesh.shape[0]
    num_comps = fmodes.shape[0]
    num_overtones = wp.f.shape[1]

    for comp_indx in range(num_comps):
        sig_i = copy.deepcopy(sig_full)
        for comp_indx2 in range(num_comps):
            if comp_indx2 != comp_indx: # and comp_indx2 != num_comps - 1:
                sig_i -= fmodes[comp_indx2, :, 0] * wave(["custom", wp.f[comp_indx2]], fmodes[comp_indx2, :, 1])

        ot = learn_waveform0(t_mesh, sig_i, np.zeros((0, N, 4)), fmodes[comp_indx], 0, num_overtones, alpha)

        overtones = np.zeros((0, 2 * num_overtones - 1))

        N_ot = ot.shape[0]

        for t_indx in range(N_ot):
            if mode_inter[comp_indx, t_indx] == 0:
                overtones = np.concatenate((overtones, ot[t_indx:t_indx + 1]), axis=0)

        ot_t = np.zeros((2 * num_overtones - 2, N_ot))
        a_i = np.ones((num_overtones, 2))
        a_i[0, 1] = 0
        for ov_indx in range(1, 2 * num_overtones - 1):
            overtone_i = overtones[:, ov_indx] / overtones[:, 0]
            ot_t[ov_indx - 1] = overtone_i
            bin_num = int((np.max(overtone_i) - np.min(overtone_i)) / 0.002) + 1
            hist, edges = np.histogram(overtone_i, bins=bin_num)

            #if ov_indx == 2 or ov_indx == 4:
            #    plt.plot(t_mesh, overtone_i)
            #    plt.show()

            #    plt.hist(overtone_i, bins=bin_num)
            #    plt.show()

            hist_max = np.argmax(hist)

            if hist[hist_max] / np.sum(hist) > 0.05:
                overtone_i = overtone_i[overtone_i <= edges[hist_max + 1]]
                overtone_i = overtone_i[overtone_i >= edges[hist_max]]
                if ov_indx < num_overtones:
                    a_i[ov_indx, 0] = np.mean(overtone_i)
                else:
                    a_i[ov_indx - num_overtones + 1, 1] = np.mean(overtone_i)

            else:
                if ov_indx < num_overtones:
                    a_i[ov_indx, 0] = 0
                else:
                    a_i[ov_indx - num_overtones + 1, 1] = 0
        print(a_i)
        wp.f[comp_indx] = a_i
    #np.save("ot_t.npy", ot_t)
    return wp


def semimanual_maxpool_peel2(signal, wave_p=0, alpha=25, t_mesh=np.zeros((1)), thr=0.005, thr_en=0.1, ref_fin=False):
    #t_mesh: time mesh of signal (evenly spaced increments)
    #signal: the signal to be decomposed into modes
    #alpha: the width of the Gaussian window
    #wave_params: base waveform parameters (see "wave" function for details)
    #thresh: the threshold to stop the opt final loop (\epsilon_1 in Alg 6)
    #thresh_en: the threshold to identify \omega_low as in eq 7.15 (it is a factor of the maximum energy,
    #   so 0.05 represents 5% of the max energy)
    #refine_final: True attempts to refine to near machine precision, False does not attempt
    #   only select True if the modes are non-intersecting/vanishing, there is no noise,
    #   and the waveforms are "near trigonometric," e.g. tri wave is ok but ekg is not
    #Returns fmodes which is of the shape (number of submodes, signal size, 4)
    #   axis 2 = 0 contains amp, 1 contains phase, 2 freq, 3 low and high stops

    N = signal.shape[0]
    if t_mesh.shape[0] != N:
        t_mesh = np.arange((N)) * 2.0 / N
        t_mesh -= np.mean(t_mesh)
    t_mesh -= np.mean(t_mesh)
    ss = t_mesh[1] - t_mesh[0]
    num_ov = 0
    om_mesh = make_omega_mesh(t_mesh, alpha)
    if isinstance(wave_p, int):
        wave_params = ["cos", 0]
    else:
        wave_params = wave_p
    if wave_params[0] == "unk":
        num_ov = wave_params[1]

    original_signal = copy.deepcopy(signal)

    fmodes = np.zeros((0, N, 4)) #axis 2 = 0 contains amp, 1 contains phase, 2 freq, 3 low and high stops
    tmodes = np.zeros((0, N, 2, 2))

    class WP_all:
        def __init__(self, wave_params, final, temp):
            self.w = wave_params
            self.f = final
            self.t = temp

    wp = WP_all(wave_params, np.zeros((0, num_ov, 2)), np.zeros((0, num_ov, 2)))

    E = compute_E_fast(t_mesh, om_mesh, wp, signal, fmodes, tmodes, 0, alpha)

    E_border = compute_E_border(t_mesh, om_mesh, alpha)
    E_thresh = thr_en * np.max(E)

    cond = True

    while cond:
        signal = copy.deepcopy(original_signal)
        for mode_i in range(fmodes.shape[0]):
            wave_params_i = wp.w
            if wp.w[0] == "unk":
                wave_params_i = ["custom", wp.f[mode_i]]
            signal -= fmodes[mode_i, :, 0] * wave(wave_params_i, fmodes[mode_i, :, 1])

        omega_cut = np.zeros((N, 2), dtype=np.int)

        print(E_thresh)

        modes_c_full, E = get_raw_mode_fast(t_mesh, om_mesh, wp, signal, fmodes, tmodes, E_thresh, E_border, alpha, 0.5)

        if np.count_nonzero(modes_c_full[:, 0, 0]) == 0: cond = False
        else:
            num_modes = 0

            modes_c = mode_consolidate(t_mesh, om_mesh, wp, signal, tmodes, modes_c_full, alpha)
            print(modes_c.shape[0])

            if modes_c.shape[0] == 0: cond = False
            else:
                cond_manual = False

                fm = np.zeros((0, N, 4))

                if tmodes.shape[0] == 0:
                    for mode_i in range(modes_c.shape[0]):
                        low_stop = np.min(np.nonzero(modes_c[mode_i, :, 0, 0])[0])
                        high_stop = np.max(np.nonzero(modes_c[mode_i, :, 0, 0])[0]) + 1

                        omega_low = modes_c[mode_i, low_stop, 0, 0]
                        omega_high = modes_c[mode_i, high_stop - 1, 0, 0]

                        buffer_low = int(2 * math.pi / (omega_low * ss))
                        buffer_high = int(2 * math.pi / (omega_high * ss))

                        amp_i = np.sqrt(modes_c[mode_i, :, 1, 0] ** 2 + modes_c[mode_i, :, 1, 1] ** 2)
                        amp_max = np.max(amp_i)
                        amp_low = np.min(amp_i[low_stop:low_stop + buffer_low])
                        amp_high = np.min(amp_i[high_stop - buffer_high:high_stop])

                        if low_stop < buffer_low * 3 or amp_low / amp_max < 0.1:
                            if high_stop > N - buffer_high * 3 or amp_high / amp_max < 0.1:
                                cond_manual = True
                                num_modes += 1
                                modes_i = modes_c[mode_i:mode_i + 1]
                                rot = np.zeros((0))
                                final_modes_i = final_mode_group3(t_mesh, wp, signal, modes_i, tmodes, rot, alpha)

                                if wave_params[0] != "unk":
                                    final_modes_i[0, :, 0] /= fund_coeff(wave_params)
                                else:
                                    c = learn_waveform(t_mesh, signal, fmodes, modes_i, wp, num_ov, alpha)
                                    wp.f = np.concatenate((wp.f, c.reshape((1, num_ov, 2))), axis=0)

                                fm = np.concatenate((fm, final_modes_i), axis=0)
                if fm.shape[0] > 0:
                    mode_cov = np.zeros((N), dtype=np.int)
                    mode_len = np.zeros((fm.shape[0]))

                    for i in range(fm.shape[0]):
                        low_stop = int(fm[i, 0, 3])
                        high_stop = int(fm[i, 1, 3])
                        mode_cov[low_stop:high_stop + 1] += 1
                        mode_len[i] = high_stop - low_stop

                    if np.max(mode_cov) == 1:
                        fmodes = np.concatenate((fmodes, fm), axis=0)

                    else:
                        fmodes = np.concatenate((fmodes, fm[np.argmax(mode_len)].reshape((1, N, 4))), axis=0)

                if not cond_manual:
                    modes_freq = np.concatenate((tmodes[:, :, 0, 0], modes_c[:, :, 0, 0]), axis=0)
                    modes_full = np.concatenate((tmodes, modes_c), axis=0)

                    print("The first " + str(tmodes.shape[0]) + " modes are old and the next " + str(
                        modes_c.shape[0]) + " modes are new.")

                    user_input, temp_modes_indx = get_input_modedivide(t_mesh, om_mesh, modes_freq, E,
                                                                       "Determine which raw sub-modes to group")

                    tmodes = np.zeros((temp_modes_indx.shape[0], N, 2, 2))
                    wp.t = np.zeros((0, num_ov, 2))
                    for mode_i in range(temp_modes_indx.shape[0]):
                        tmodes[mode_i] = modes_full[temp_modes_indx[mode_i]]
                        if wave_params[0] != "unk":
                            tmodes[mode_i, :, 1] /= fund_coeff(wave_params)
                        else:
                            c = learn_waveform(t_mesh, signal, fmodes, tmodes[mode_i:mode_i + 1], wp, num_ov, alpha)
                            wp.t = np.concatenate((wp.t, c.reshape((1, num_ov, 2))), axis=0)

                    num_modes = len(user_input)
                    num_temp_modes = tmodes.shape[0]

                    if num_modes + num_temp_modes == 0: cond = False

                    elif num_modes > 0:
                        num_modes = len(user_input)

                        final_modes_i = np.zeros((num_modes, N, 4))

                        for mode_i in range(num_modes):
                            modes_i = np.zeros((user_input[mode_i].shape[0], N, 2, 2))
                            for mode_indx2 in range(user_input[mode_i].shape[0]):
                                modes_i[mode_indx2] = modes_full[user_input[mode_i][mode_indx2]]

                            rot = np.zeros((user_input[mode_i].shape[0]))
                            final_modes_i[mode_i] = final_mode_group3(t_mesh, wp, signal, modes_i, tmodes, rot, alpha)

                            if wave_params[0] != "unk":
                                final_modes_i[mode_i, :, 0] /= fund_coeff(wave_params)

                            else:
                                c = learn_waveform(t_mesh, signal, fmodes, modes_i, wave_params, num_ov, alpha)
                                wp.f = np.concatenate((wp.f, c.reshape((1, num_ov, 2))), axis=0)

                        fmodes = np.concatenate((fmodes, final_modes_i), axis=0)

                        for mode_i in range(fmodes.shape[0]):
                            fmodes[mode_i, :, 1] = phase_modtosmooth(fmodes[mode_i, :, 1])

                if cond and num_modes > 0:
                    fmodes, wp = stab_opt_final10(t_mesh, original_signal, fmodes, wp, thr, 8, alpha, alpha, 1.0, 100)

                omega_cut[:, 0] = copy.deepcopy(omega_cut[:, 1])

    for mode_i in range(fmodes.shape[0]):
        fmodes[mode_i, :, 1] = phase_modtosmooth(fmodes[mode_i, :, 1])

    if ref_fin and fmodes.shape[0] > 0:
        #TODO: work on these hyperparameters
        if wp.w[0] == "ekg":
            fmodes, wp = stab_opt_final10(t_mesh, original_signal, fmodes, wp, thr / 100.0, 8, 20, 10, 0.1, 300)
        elif wp.w[0] == "tri":
            fmodes, wp = stab_opt_final10(t_mesh, original_signal, fmodes, wp, thr / 100.0, 2, 12, 6, 0.1, 300)
        else:
            fmodes, wp = stab_opt_final10(t_mesh, original_signal, fmodes, wp, thr / 100.0, 8, 20, 10, 0.1, 300)

    return fmodes, wp

def manual_maxpool_peel2(signal, wave_p=0, alpha=25, t_mesh=np.zeros((1)), thr=0.005, thr_en=0.1, ref_fin=False):
    #t_mesh: time mesh of signal (evenly spaced increments)
    #signal: the signal to be decomposed into modes
    #alpha: the width of the Gaussian window
    #wave_params: base waveform parameters (see "wave" function for details)
    #thresh: the threshold to stop the opt final loop (\epsilon_1 in Alg 6)
    #thresh_en: the threshold to identify \omega_low as in eq 7.15 (it is a factor of the maximum energy,
    #   so 0.05 represents 5% of the max energy)
    #refine_final: True attempts to refine to near machine precision, False does not attempt
    #   only select True if the modes are non-intersecting/vanishing, there is no noise,
    #   and the waveforms are "near trigonometric," e.g. tri wave is ok but ekg is not
    #Returns fmodes which is of the shape (number of submodes, signal size, 4)
    #   axis 2 = 0 contains amp, 1 contains phase, 2 freq, 3 low and high stops


    N = signal.shape[0]
    if t_mesh.shape[0] != N:
        t_mesh = np.arange((N)) * 2.0 / N
        t_mesh -= np.mean(t_mesh)
    t_mesh -= np.mean(t_mesh)
    num_ov = 0
    om_mesh = make_omega_mesh(t_mesh, alpha)
    if isinstance(wave_p, int):
        wave_params = ["cos", 0]
    else:
        wave_params = wave_p
    if wave_params[0] == "unk":
        num_ov = wave_params[1]

    original_signal = copy.deepcopy(signal)

    fmodes = np.zeros((0, N, 4)) #axis 2 = 0 contains amp, 1 contains phase, 2 freq, 3 low and high stops
    tmodes = np.zeros((0, N, 2, 2))

    class WP_all:
        def __init__(self, wave_params, final, temp):
            self.w = wave_params
            self.f = final
            self.t = temp

    wp = WP_all(wave_params, np.zeros((0, num_ov, 2)), np.zeros((0, num_ov, 2)))

    E = compute_E_fast(t_mesh, om_mesh, wp, signal, fmodes, tmodes, 0, alpha)

    E_border = compute_E_border(t_mesh, om_mesh, alpha)
    E_thresh = thr_en * np.max(E)

    cond = True

    while cond:
        signal = copy.deepcopy(original_signal)
        for mode_indx in range(fmodes.shape[0]):
            wave_params_i = wp.w
            if wp.w[0] == "unk":
                wave_params_i = ["custom", wp.f[mode_indx]]
            signal -= fmodes[mode_indx, :, 0] * wave(wave_params_i, fmodes[mode_indx, :, 1])

        omega_cut = np.zeros((N, 2), dtype=np.int)

        print(E_thresh)

        modes_c_full, E = get_raw_mode_fast(t_mesh, om_mesh, wp, signal, fmodes, tmodes, E_thresh, E_border, alpha, 0.5)

        if np.count_nonzero(modes_c_full[:, 0, 0]) == 0: cond = False
        else:
            modes_c = mode_consolidate(t_mesh, om_mesh, wp, signal, tmodes, modes_c_full, alpha)

            if modes_c.shape[0] == 0: cond = False
            else:
                modes_freq = np.concatenate((tmodes[:, :, 0, 0], modes_c[:, :, 0, 0]), axis=0)
                modes_full = np.concatenate((tmodes, modes_c), axis=0)

                print("The first " + str(tmodes.shape[0]) + " modes are old and the next " + str(
                    modes_c.shape[0]) + " modes are new.")

                user_input, temp_modes_indx = get_input_modedivide(t_mesh, om_mesh, modes_freq, E,
                                                                   "Determine which raw sub-modes to group")

                tmodes = np.zeros((temp_modes_indx.shape[0], N, 2, 2))
                wp.t = np.zeros((0, num_ov, 2))
                for mode_indx in range(temp_modes_indx.shape[0]):
                    tmodes[mode_indx] = modes_full[temp_modes_indx[mode_indx]]
                    if wave_params[0] != "unk":
                        tmodes[mode_indx, :, 1] /= fund_coeff(wave_params)
                    else:
                        c = learn_waveform(t_mesh, signal, fmodes, tmodes[mode_indx:mode_indx + 1], wp, num_ov, alpha)
                        wp.t = np.concatenate((wp.t, c.reshape((1, num_ov, 2))), axis=0)

                num_modes = len(user_input)
                num_temp_modes = tmodes.shape[0]

                if num_modes + num_temp_modes == 0: cond = False

                elif num_modes > 0:
                    num_modes = len(user_input)

                    final_modes_i = np.zeros((num_modes, N, 4))

                    for mode_indx in range(num_modes):
                        modes_i = np.zeros((user_input[mode_indx].shape[0], N, 2, 2))
                        for mode_indx2 in range(user_input[mode_indx].shape[0]):
                            modes_i[mode_indx2] = modes_full[user_input[mode_indx][mode_indx2]]

                        rot = np.zeros((user_input[mode_indx].shape[0]))
                        final_modes_i[mode_indx] = final_mode_group3(t_mesh, wp, signal, modes_i, tmodes, rot, alpha)

                        if wave_params[0] != "unk":
                            final_modes_i[mode_indx, :, 0] /= fund_coeff(wave_params)

                        else:
                            c = learn_waveform(t_mesh, signal, fmodes, modes_i, wave_params, num_ov, alpha)
                            wp.f = np.concatenate((wp.f, c.reshape((1, num_ov, 2))), axis=0)

                    fmodes = np.concatenate((fmodes, final_modes_i), axis=0)

                    for mode_indx in range(fmodes.shape[0]):
                        fmodes[mode_indx, :, 1] = phase_modtosmooth(fmodes[mode_indx, :, 1])

                if cond and num_modes > 0:
                    fmodes, wp = stab_opt_final10(t_mesh, original_signal, fmodes, wp, thr, 8, alpha, alpha, 1.0, 100)

                omega_cut[:, 0] = copy.deepcopy(omega_cut[:, 1])

    for mode_indx in range(fmodes.shape[0]):
        fmodes[mode_indx, :, 1] = phase_modtosmooth(fmodes[mode_indx, :, 1])

    if ref_fin and fmodes.shape[0] > 0:
        #TODO: work on these hyperparameters
        if wp.w[0] == "ekg":
            fmodes, wp = stab_opt_final10(t_mesh, original_signal, fmodes, wp, thr / 100.0, 8, 20, 10, 0.1, 300)
        elif wp.w[0] == "tri":
            fmodes, wp = stab_opt_final10(t_mesh, original_signal, fmodes, wp, thr / 100.0, 2, 12, 6, 0.1, 300)
        else:
            fmodes, wp = stab_opt_final10(t_mesh, original_signal, fmodes, wp, thr / 100.0, 8, 20, 10, 0.1, 300)

    return fmodes, wp


def rot_check(micloc, low_stop, high_stop):
    #checks whether the micro-local KMD estimate has no rotations
    bad_sin_flip = np.zeros((micloc.shape[0]))

    for t_indx in range(low_stop, high_stop - 1):
        if micloc[t_indx, 1] * micloc[t_indx + 1, 1] <= 0 and micloc[t_indx, 0] < 0:
            bad_sin_flip[t_indx] = 1

    if np.sum(bad_sin_flip) == 0: return True
    else: return False

def opt_final10(t_mesh, sig_full, fmodes, wp, thresh, win_size, alpha, min_alpha, check_freq, max_iter):
    #win_size: the window size of the savgol filter as a multiple of alpha/om in time space
    #min_alpha: alpha is reduced in the loop to a minimum alpha
    #check_freq: the frequency (relative to 2*pi/om) that micro local KMD refinements are estimated (cubic splines are used to fill gaps)
    #max_iter is the maximum number of iterations to run the loop

    #if the loop is unstable and yields a phase/freq that are not consistent, it returns success=False
    #otherwise, returns a refined fmodes (identified modes) and success=True
    #TODO: Is it possible to replace the savgol filter with something better?
    #TODO: There is a mysterious (extremely) minor bug which makes recovery of mirror images differ (extremely) slightly
    N = t_mesh.shape[0]
    ss = t_mesh[1] - t_mesh[0]
    num_comps = fmodes.shape[0]
    fmodes_copy = copy.deepcopy(fmodes)
    orig_modes = copy.deepcopy(fmodes)
    freq_orig = copy.deepcopy(fmodes[:, :, 2])
    stop_orig = copy.deepcopy(fmodes[:, 0:2, 3])
    orig_sig_full = copy.deepcopy(sig_full)
    num_iter_i = max_iter
    i = 0
    cond2 = True
    success = True

    wp_orig = copy.deepcopy(wp)

    if wp.w[0] != "unk":
        c1 = fund_coeff(wp.w)
    else:
        c1 = 1

    mode_inter = np.zeros((num_comps, N))

    for mode_indx in range(num_comps):
        for mode_indx2 in range(num_comps):
            if mode_indx > mode_indx2:
                freq_sim = np.abs(np.log2(freq_orig[mode_indx, :] / freq_orig[mode_indx2, :]))
                #plt.plot(freq_sim)
                #plt.show()
                for t_i in range(N):
                    if freq_sim[t_i] < np.log2(1.2):
                        mode_inter[mode_indx, t_i] = 1
                        mode_inter[mode_indx2, t_i] = 1

    print("Opt Final progress:")
    while cond2 and i <= num_iter_i:
        if i % 5 == 0: print(i)
        if i % 10 == 0 and i > 10 and alpha > min_alpha: alpha -= 1

        if i % 10 == 0 and i > 0:
            if wp.w[0] == "unk":
                wp = learn_waveform_final(t_mesh, orig_sig_full, fmodes, mode_inter, wp, alpha)

        for mode_i in range(num_comps):
            ls = int(fmodes[mode_i, 0, 3])
            hs = int(fmodes[mode_i, 1, 3])


            omega_low = fmodes[mode_i, ls, 2]
            omega_high = fmodes[mode_i, hs - 1, 2]

            low_stop2 = int(ls + 12 * math.pi / (omega_low * ss))
            high_stop2 = int(hs - 12 * math.pi / (omega_high * ss))

            sig_comp = np.zeros((N))
            for comp_indx2 in range(num_comps):
                if wp.w[0] == "unk":
                    wave_params = ["custom", wp.f[comp_indx2]]
                else:
                    wave_params = wp.w
                sig_comp += fmodes[comp_indx2, :, 0] * wave(wave_params, fmodes[comp_indx2, :, 1])

            sig_comp = sig_full - sig_comp

            if wp.w[0] == "unk":
                wave_params = ["custom", wp.f[mode_i]]
            else:
                wave_params = wp.w

            sig_comp2 = sig_comp + fmodes[mode_i, :, 0] * fund_wave(wave_params, fmodes[mode_i, :, 1])


            amp_i = copy.deepcopy(fmodes[mode_i, :, 0])
            theta_i = copy.deepcopy(fmodes[mode_i, :, 1])

            micloc_wave = np.zeros((N, 2))
            micloc_wave2 = np.zeros((N, 3))

            amp_max = np.max(fmodes[mode_i, :, 0])

            rots = np.random.uniform(0, 1)
            counter = 0

            for t_i in range(ls, hs):
                #if amp_i[t_indx] > 0.0 * np.max(amp_i):  # adjust threshold
                t0 = t_mesh[t_i]
                om = fmodes[mode_i, t_i, 2]

                if t_i < low_stop2 or t_i > high_stop2:
                    micloc_wave[t_i] = micloc_KMD_final(t_mesh, sig_comp2, 2, t_i, om, theta_i, alpha)[0]
                    micloc_wave[t_i] /= fund_coeff(wave_params)

                else:
                    rots += om * ss
                    if int(rots / check_freq) != int((rots - om * ss) / check_freq):
                        micloc_wave2[counter, :2] = micloc_KMD_final(t_mesh, sig_comp2, 2, t_i, om, theta_i, alpha)[0]
                        micloc_wave2[counter, :2] /= fund_coeff(wave_params)
                        micloc_wave2[counter, 2] = t0
                        counter += 1

            micloc_wave2 = micloc_wave2[:counter]

            t_mesh_i = micloc_wave2[:, 2]
            sin_int = micloc_wave2[:, 1]
            cos_int = micloc_wave2[:, 0]

            sin_int_spl = CubicSpline(t_mesh_i, sin_int)
            cos_int_spl = CubicSpline(t_mesh_i, cos_int)

            micloc_wave[low_stop2:high_stop2 + 1, 1] = sin_int_spl(t_mesh[low_stop2:high_stop2 + 1])
            micloc_wave[low_stop2:high_stop2 + 1, 0] = cos_int_spl(t_mesh[low_stop2:high_stop2 + 1])

            micloc_wave, ls, hs = mode_clip(t_mesh, freq_orig[mode_i], fmodes[mode_i, :, 2], micloc_wave)

            if rot_check(micloc_wave, ls, hs - 1) == True:
                fmodes[mode_i, ls:hs, 0] = np.sqrt(micloc_wave[ls:hs, 0] ** 2 + micloc_wave[ls:hs, 1] ** 2)
                fmodes[mode_i, ls:hs, 1] += 0.5 * np.arctan2(-micloc_wave[ls:hs, 1], micloc_wave[ls:hs, 0])

            else:
                print(False)
                fmodes[mode_i, ls:hs, 0] = np.sqrt(micloc_wave[ls:hs, 0] ** 2 + micloc_wave[ls:hs, 1] ** 2)
                dphase_i = np.arctan2(-micloc_wave[ls:hs, 1], micloc_wave[ls:hs, 0])

                dphase_i = phase_modtosmooth(dphase_i)
                fmodes[mode_i, ls:hs, 1] += dphase_i

                fmodes[mode_i, ls:hs, 1] -= (fmodes[mode_i, 0, 1] // (2 * math.pi)) * 2 * math.pi

            amp_low = fmodes[mode_i, ls, 0]
            amp_high = fmodes[mode_i, hs - 1, 0]

            om = np.mean(fmodes[mode_i, ls:hs, 2])
            wl = int(win_size * alpha / (om * ss)) #window length
            if wl > (hs - ls) // 2:
                wl = (hs - ls) // 2

            wl = 2 * (wl // 2) + 1
            po = 3 #poly order

            fmodes[mode_i, ls:hs, 1] += 0.5 * scisig.savgol_filter(fmodes[mode_i, ls:hs, 1], wl, po)
            fmodes[mode_i, ls:hs, 0] += 0.5 * np.exp(-i / 50) * scisig.savgol_filter(fmodes[mode_i, ls:hs, 0], wl, po)

            fmodes[mode_i, ls:hs, 1] /= 1.5
            fmodes[mode_i, ls:hs, 0] /= 1 + 0.5 * np.exp(-i / 50)

            if np.min(fmodes[mode_i, :, 0]) < 0:
                fmodes[mode_i, :, 0] = np.maximum(0, fmodes[mode_i, :, 0])

            border_fudge_low = ((t_mesh[ls:ls + wl] - t_mesh[ls + wl]) / (t_mesh[ls] - t_mesh[ls + wl])) ** 2
            fmodes[mode_i, ls:ls + wl, 0] -= (fmodes[mode_i, ls, 0] - amp_low) * border_fudge_low
            fmodes[mode_i, :ls, 0:3] = 0

            border_fudge_high = ((t_mesh[hs - wl:hs] - t_mesh[hs - wl]) / (t_mesh[hs - 1] - t_mesh[hs - wl])) ** 2
            fmodes[mode_i, hs - wl:hs, 0] -= (fmodes[mode_i, hs - 1, 0] - amp_high) * border_fudge_high
            fmodes[mode_i, hs:, 0:3] = 0

            fmodes[mode_i, ls:hs - 1, 2] = (fmodes[mode_i, ls + 1:hs, 1] - fmodes[mode_i, ls:hs - 1, 1]) / ss
            fmodes[mode_i, hs - 1, 2] = fmodes[mode_i, hs - 2, 2]
            fmodes[mode_i, ls:hs, 2] = scisig.savgol_filter(fmodes[mode_i, ls:hs, 2], wl, po)

            freq_dev = fmodes[mode_i, ls:hs, 2] / freq_orig[mode_i, ls:hs]

            amp_max = np.max(fmodes[mode_i, :, 0])

            winL = int(6 * math.pi / (fmodes[mode_i, ls, 2] * ss))
            wind_high = int(6 * math.pi / (fmodes[mode_i, hs - 1, 2] * ss))


            for t_i in range(ls, hs):
                if freq_dev[t_i - ls] > 1.1:
                    fmodes[mode_i, t_i, 2] = 1.1 * freq_orig[mode_i, t_i]
                if freq_dev[t_i - ls] < 1 / 1.1:
                    fmodes[mode_i, t_i, 2] = freq_orig[mode_i, t_i] / 1.1

                if fmodes[mode_i, t_i, 0] < 0.1 * amp_max:
                    freq_dev[t_i - ls] = 1
                if t_i < ls + winL or t_i > hs - wind_high:
                    freq_dev[t_i - ls] = 1


            max_freq_dev = np.max(freq_dev)
            min_freq_dev = np.min(freq_dev)

            if max_freq_dev > 1.5 or min_freq_dev < 1 / 1.5:
                success = False
                cond2 = False
                print(max_freq_dev, min_freq_dev)
                plt.plot(fmodes[mode_i, :, 2])
                plt.show()

                plt.plot(freq_dev)
                plt.show()
                return orig_modes, wp_orig, success

            #The next 4 if statements control the edges of the mode if necessary
            if stop_orig[mode_i, 0] == 0:
                omega_low = freq_orig[mode_i, 0]
                winL = int(alpha / (omega_low * ss))
                if ls != 0:
                    fit_low0 = np.polyfit(t_mesh[ls:ls + winL], fmodes[mode_i, ls:ls + winL, 0], 1)
                    fmodes[mode_i, 0:ls, 0] = 0
                    for deg in range(1 + 1):
                        fmodes[mode_i, 0:ls, 0] += fit_low0[1 - deg] * t_mesh[:ls] ** deg
                    fmodes[mode_i, :ls, 0] += (-ss * fit_low0[0]) + (fmodes[mode_i, ls, 0] - fmodes[mode_i, ls - 1, 0])


                    fit_low1 = np.polyfit(t_mesh[ls:ls + winL], fmodes[mode_i, ls:ls + winL, 1], 2)
                    fmodes[mode_i, 0:ls, 1] = 0
                    for deg in range(2 + 1):
                        fmodes[mode_i, 0:ls, 1] += fit_low1[2 - deg] * t_mesh[:ls] ** deg
                    d_th = fmodes[mode_i, ls, 1] - fmodes[mode_i, ls - 1, 1]
                    fmodes[mode_i, :ls, 1] += -ss * (fit_low1[1] + 2 * fit_low1[0] * t_mesh[ls]) + d_th


                    fit_low2 = np.polyfit(t_mesh[ls:ls + winL], fmodes[mode_i, ls:ls + winL, 2], 1)
                    fmodes[mode_i, 0:ls, 2] = 0
                    for deg in range(1 + 1):
                        fmodes[mode_i, 0:ls, 2] += fit_low2[1 - deg] * t_mesh[:ls] ** deg
                    fmodes[mode_i, :ls, 2] += (-ss * fit_low2[0]) + (fmodes[mode_i, ls, 2] - fmodes[mode_i, ls - 1, 2])

                if not 1 / 1.1 < fmodes[mode_i, winL, 0] / fmodes[mode_i, 0, 0] < 1.1: #change to fixed difference maybe
                    fit_low = np.polyfit(t_mesh[winL:2 * winL], fmodes[mode_i, winL:2 * winL, 0], 1)
                    fmodes[mode_i, 0:winL, 0] = 0
                    for deg in range(1 + 1):
                        fmodes[mode_i, 0:winL, 0] += fit_low[1 - deg] * t_mesh[:winL] ** deg
                fmodes[mode_i, 0, 3] = 0

            if stop_orig[mode_i, 0] != 0:
                orig_low_stop = int(stop_orig[mode_i, 0])
                omega_low = freq_orig[mode_i, orig_low_stop]
                winL = int(2 * math.pi / (omega_low * ss))

                fit_low = np.polyfit(t_mesh[ls:ls + winL], fmodes[mode_i, ls:ls + winL, 0], 1)
                if fit_low[0] > 0:
                    ls2 = max(0, ls - int(2 * fmodes[mode_i, ls, 0] / (ss * fit_low[0])))
                    t_min = t_mesh[ls] - 2 * fmodes[mode_i, ls, 0] / fit_low[0]

                    fmodes[mode_i, ls2:ls, 0] = fmodes[mode_i, ls, 0]
                    fmodes[mode_i, ls2:ls, 0] *= ((t_mesh[ls2:ls] - t_min) / (t_mesh[ls] - t_min)) ** 2
                    fmodes[mode_i, :ls, 1] = fmodes[mode_i, ls, 2] * (t_mesh[:ls] - t_mesh[ls]) + fmodes[mode_i, ls, 1]
                    fmodes[mode_i, :ls, 2] = fmodes[mode_i, ls, 2]
                    fmodes[mode_i, 0, 3] = ls2

                else:
                    ls2 = max(0, ls - winL)
                    t_min = t_mesh[ls] - winL * ss

                    fmodes[mode_i, ls2:ls, 0] = fmodes[mode_i, ls, 0]
                    fmodes[mode_i, ls2:ls, 0] *= ((t_mesh[ls2:ls] - t_min) / (t_mesh[ls] - t_min)) ** 2
                    fmodes[mode_i, :ls, 1] = fmodes[mode_i, ls, 2] * (t_mesh[:ls] - t_mesh[ls]) + fmodes[mode_i, ls, 1]
                    fmodes[mode_i, :ls, 2] = fmodes[mode_i, ls, 2]
                    fmodes[mode_i, 0, 3] = ls2

            if stop_orig[mode_i, 1] == N:
                omega_high = freq_orig[mode_i, -1]
                winH = int(alpha / (omega_high * ss))
                if hs != N:
                    fit_high0 = np.polyfit(t_mesh[hs - winH:hs], fmodes[mode_i, hs - winH:hs, 0], 1)
                    fmodes[mode_i, hs:, 0] = 0
                    for deg in range(1 + 1):
                        fmodes[mode_i, hs:, 0] += fit_high0[1 - deg] * t_mesh[hs:] ** deg
                    fmodes[mode_i, hs:, 0] -= (-ss * fit_high0[0]) + (fmodes[mode_i, hs, 0] - fmodes[mode_i, hs - 1, 0])

                    fh1 = np.polyfit(t_mesh[hs - winH:hs], fmodes[mode_i, hs - winH:hs, 1], 2)
                    fmodes[mode_i, hs:, 1] = 0
                    for deg in range(2 + 1):
                        fmodes[mode_i, hs:, 1] += fh1[2 - deg] * t_mesh[hs:] ** deg
                    d_th = fmodes[mode_i, hs, 1] - fmodes[mode_i, hs - 1, 1]
                    fmodes[mode_i, hs:, 1] -= -ss * (fh1[1] + 2 * fh1[0] * t_mesh[hs - 1]) + d_th


                    fit_high2 = np.polyfit(t_mesh[hs - winH:hs], fmodes[mode_i, hs - winH:hs, 2], 1)
                    fmodes[mode_i, hs:, 2] = 0
                    for deg in range(1 + 1):
                        fmodes[mode_i, hs:, 2] += fit_high2[1 - deg] * t_mesh[hs:] ** deg
                    fmodes[mode_i, hs:, 2] -= (-ss * fit_high2[0]) + (fmodes[mode_i, hs, 2] - fmodes[mode_i, hs - 1, 2])

                if not 1 / 1.1 < fmodes[mode_i, N - winH, 0] / fmodes[mode_i, N - 1, 0] < 1.1:
                    fit_high = np.polyfit(t_mesh[N - 2 * winH:N - winH], fmodes[mode_i, N - 2 * winH:N - winH, 0], 1)
                    fmodes[mode_i, N - winH:N, 0] = 0
                    for deg in range(1 + 1):
                        fmodes[mode_i, N - winH:N, 0] += fit_high[1 - deg] * t_mesh[N - winH:N] ** deg
                fmodes[mode_i, 1, 3] = N

            if stop_orig[mode_i, 1] != N:
                orig_high_stop = int(stop_orig[mode_i, 1])
                omega_high = freq_orig[mode_i, orig_high_stop - 1]
                wind_high = int(2 * math.pi / (omega_high * ss))
                fit_high = np.polyfit(t_mesh[hs - wind_high:hs], fmodes[mode_i, hs - wind_high:hs, 0], 1)
                if fit_high[0] < 0:
                    fh0 = -fit_high[0]
                    hs2 = min(N, hs + int(2 * fmodes[mode_i, hs - 1, 0] / (ss * fh0)))
                    t_max = t_mesh[hs] + 2 * fmodes[mode_i, hs - 1, 0] / fh0

                    fmodes[mode_i, hs:hs2, 0] = fmodes[mode_i, hs - 1, 0]
                    fmodes[mode_i, hs:hs2, 0] *= ((t_mesh[hs:hs2] - t_max) / (t_mesh[hs - 1] - t_max)) ** 2
                    old_th = fmodes[mode_i, hs - 1, 1]
                    fmodes[mode_i, hs:, 1] = fmodes[mode_i, hs - 1, 2] * (t_mesh[hs:] - t_mesh[hs - 1]) + old_th
                    fmodes[mode_i, hs:, 2] = fmodes[mode_i, hs - 1, 2]
                    fmodes[mode_i, 1, 3] = hs2

                else:
                    hs2 = max(0, hs + wind_high)
                    t_max = t_mesh[hs] + wind_high * ss

                    mesh_buf = ((t_mesh[hs:hs2] - t_max) / (t_mesh[hs - 1] - t_max)) ** 2
                    fmodes[mode_i, hs:hs2, 0] = fmodes[mode_i, hs - 1, 0] * mesh_buf
                    old_th = fmodes[mode_i, hs - 1, 1]
                    fmodes[mode_i, hs:, 1] = fmodes[mode_i, hs - 1, 2] * (t_mesh[hs:] - t_mesh[hs]) + old_th
                    fmodes[mode_i, hs:, 2] = fmodes[mode_i, hs - 1, 2]
                    fmodes[mode_i, 1, 3] = hs2

        if cond2:
            if i == 10:
                comp_data_iter = copy.deepcopy(fmodes[:, :, 0:2])
            if i > 10 and i % 10 == 0:
                max_change = 0
                for comp_indx3 in range(fmodes.shape[0]):
                    ls = int(fmodes[comp_indx3, 0, 3])
                    hs = int(fmodes[comp_indx3, 1, 3])

                    mode_i0 = fmodes[comp_indx3, ls:hs, 0] * np.cos(fmodes[comp_indx3, ls:hs, 1])
                    mode_i1 = fmodes[comp_indx3, ls:hs, 0] * np.sin(fmodes[comp_indx3, ls:hs, 1])

                    old_mode_i0 = comp_data_iter[comp_indx3, ls:hs, 0] * np.cos(comp_data_iter[comp_indx3, ls:hs, 1])
                    old_mode_i1 = comp_data_iter[comp_indx3, ls:hs, 0] * np.sin(comp_data_iter[comp_indx3, ls:hs, 1])

                    max_change = max(0, math.sqrt(np.max((mode_i0 - old_mode_i0) ** 2 + (mode_i1 - old_mode_i1) ** 2)))

                if i == 20:
                    old_max_change = 2 * max_change
                print(max_change)
                if max_change < thresh or i == num_iter_i:
                    cond2 = False

                if max_change >= old_max_change * 1.01 and alpha < min_alpha + 1:
                    cond2 = False
                    fmodes = copy.deepcopy(fmodes_copy)

                comp_data_iter = copy.deepcopy(fmodes[:, :, 0:2])
                old_max_change = copy.deepcopy(max_change)
                fmodes_copy = copy.deepcopy(fmodes)

        i += 1


    return fmodes, wp, success

def stab_opt_final10(t_mesh, sig_full, fmodes, wp, thresh, win_size, alpha, min_alpha, check_freq, max_iter):
    #Runs opt_final10 with given window size, if the loop is unstable, it increases win_size of the savgol filter (which leads to stability in experiments)
    success = False
    orig_modes = copy.deepcopy(fmodes)

    while not success:
        fmodes2, wp, success = opt_final10(t_mesh, sig_full, orig_modes, wp, thresh, win_size, alpha, min_alpha, check_freq, max_iter)
        orig_modes = copy.deepcopy(fmodes)
        win_size += 1

    return fmodes2, wp
