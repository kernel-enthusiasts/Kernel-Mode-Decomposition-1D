# Kernel Mode Decomposition for 1D signals

This code is an implementation of the mode decomposition algorithm detailed in https://arxiv.org/abs/1907.08592.  Signals which are the sums noise and a certain number of nearly periodic modes, meaning each mode is of form <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a(t)&space;y(\theta(t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a(t)&space;y(\theta(t))" title="a(t) y(\theta(t))" /></a>, where y is some periodic function, called the base waveform, and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;a(t),&space;\theta(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;a(t),&space;\theta(t)" title="a(t), \theta(t)" /></a> are relatively slowly varying on timescales of order <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\frac{1}{\theta'(t)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{1}{\theta'(t)}" title="\frac{1}{\theta'(t)}" /></a>.  Signals not of this form are outside the scope of this project.

# Installation

Anaconda 4.7.12 and Python 3.7 was used to run the project.  All functions are contained in file KMD_lib.py and all other .py files are examples on usage

# Usage

The main decomposition algorithm is implemented in two slightly different ways in the functions *semimanual_maxpool_peel2* and *manual_maxpool_peel2*.  The inputs of both functions are:

**signal**: The signal to be decomposed into modes
**waveform** (optional): The base waveform of the signal.  This should be a two entry list with the first entry being "cos" (cosine), "tri" (triangle wave), "ekg" (synthetic EKG-like wave), "custom" (custom waveform given with Fourier series), or "unk" (unknown waveform).  If first entry is "cos", "tri", or "ekg", the second entry is irrelevant with default value "0".  If it is "custom", set the second entry to be a numpy array, a, of shape (N, 2).  Then the base waveform is of form 

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{k}&space;a[k,&space;0]&space;\cos((k&plus;1)&space;t)&space;&plus;&space;a[k,1]&space;\sin((k&space;&plus;&space;1)t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{k}&space;a[k,&space;0]&space;\cos((k&plus;1)&space;t)&space;&plus;&space;a[k,1]&space;\sin((k&space;&plus;&space;1)t)" title="\sum_{k} a[k, 0] \cos((k+1) t) + a[k,1] \sin((k + 1)t)" /></a>

The current implementation requires (WLOG) that a[0, 0] = 1 and a[0, 1] = 0.  If the first entry is "unk", set the second entry to be the desired number of overtones the algorithm estimates.

**alpha** (optional): the width of the Gaussian window
**t_mesh** (optional): the time mesh of the signal (must be evenly spaced).
**thresh** (optional): the threshold to stop the opt final loop (\epsilon_1 in Alg 6)
**thresh_en** (optional): the threshold to identify \omega_low as in eq 7.15 (it is a factor of the maximum energy, so 0.05 represents 5% of the max energy)
**ref_fin** (optional): True attempts to refine to near machine precision, False does not attempt only select True if the modes are non-intersecting/vanishing and there is no noise.  Also, this slows the mode decomposition greatly.

The output of this function is:

**fmodes**: An numpy array of shape (number of submodes, signal size, 4) containing the identified modes.  Note that axis 2 = 0 contains amp, 1 contains phase, 2 freq, 3 low and high stops (bounds of support of the signal).

The only difference is the prior makes an attempt to identify full modes (from the mode segments) and automatically peels them.  If no mode segments are identified as full modes, then the mode segments are passed to the user to either be combined into full modes, passed to the next iteration (if identified mode segments don't approximately span a full mode), or disregarded (if segments are artifacts of noise or a frequency intersection).  The *manual_maxpool_peel2* function will be passed to the user for judgement.

The first prompt in this user input is: 

>Input mode segments to add to mode 1 (if done with mode: Next, if done with all modes: Done):

The proper input those of those mode segments comprising of any full mode.  The user should input *mode segment a* then enter, *mode segment b* then enter, and so on.  When finished inputting these modes, *Next* should be inputted if there are more mode segments to be grouped and *Done* otherwise.  Note that when there are crossing frequencies, it is recommended for modes not to be grouped until after both modes have been fully identified outside of near the intersection (these segments should be all passed to the next iteration).

After this step has been completed, the modes to be passed the next iteration will be entered.  The user will see prompt:

>The following modes remain, select which ones to keep:

The user should then input the segments to pass to the next iteration in the same manner.  When done, input *Done*.

If there are no crossing frequency modes, *semimanual_maxpool_peel2* should not require any user input.

# Examples

*tri_wave.py* and *ekg_wave.py* are implementations of the algorithm on examples with base triangle and synthetic EKG-like waveforms.  Since they have no frequency intersections, they require no user input. 

*wave_learn.py* is an example of learning the waveforms of each mode.

*tri_cross_vanish_noise.py* and *tri_cross_vanish_noise2.py* are examples of triangle wave modes that have crossing frequencies, vanish, and is corrupted with white noise.  The first is the example used in the paper.  The recommended inputs are as follows:

*tri_cross_vanish_noise.py*:

Done, 0, 1, Done ... 0, 3, Next, 1, 2, Done 

Note that the two sequences only identifies the lowest two intersecting frequency modes.  The final mode is identified automatically.

*tri_cross_vanish_noise2.py*:

0, Done, 1, 3, Done ... 0, 4, Next, 1, 2, Done ...
