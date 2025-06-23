#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import numpy
from matplotlib import pyplot
import scipy.fft
from scipy.optimize import minimize
import scipy.stats
from sklearn.decomposition import FastICA, PCA

from openbci.data import read_file, get_physiology, hampel, \
    butter_bandpass_filter, movement_filter, compute_signal_magnitude, \
    spectrogram

import warnings
warnings.filterwarnings("ignore")


# # # # #
# CONSTANTS

# Excluded participants
EXCLUDED = [ \
    ]
# Included conditions.
CONDITIONS = ["disgust", "healthy", "unhealthy", "baseline"]

# SIGNAL
# Board that the signal was collected on.
BOARD = "ganglion"
# Filter values, in Hz (easiest to write value in cycles per minute, then 
# divide by 60 seconds; e.g. 0.5/60 and 10.0/60.
HIGH_PASS = 1.0 / 60
LOW_PASS = 10.0 / 60
# Duration of each segment computer in the spectrogram in seconds.
SPECTROGRAM_SEGMENT_DURATION_SECS = 60.0
# Define frequency bands relevant in EGG (values are in Hz, but are typically
# given in cpm, hence the Hz=cpm/60 notation).
BANDS = { \
    "brady": (1.0/60,  2.0/60), \
    "normo": (2.0/60,  4.0/60), \
    "tachy": (4.0/60, 10.0/60), \
    }
# Frequency range in which to check extracted channels' or components' power.
# Several ranges can be used at once, but the most important one should be 
# provided first within each tuple, as it will be used to compute peak power.
ICA_TARGET_FREQ = [BANDS["normo"]]
# Threshold for the exclusion of high values, based on the median and median
# absolute deviation across the whole recording. Any detected outliers are
# replaced by the median across the whole signal. Set to None to not do any
# exclusions.
OUTLIER_T = 5.0
# Movement filter settings.
MOVEMENT_FILTER = True
FREQ_OF_INTEREST = sum(BANDS["normo"]) / 2
# Signal-noise ratio threshold for inclusion of ICA components, or None to not
# to ICA de-noising.
SNR_THRESHOLD = 3.0
# Combine signal power by averaging across channels? (This is recommended if
# ICA de-noising is enabled.) The opposite of this is not to take separate
# channels (which happens anyway), but rather to take the channel with the
# strongest normogastric peak to noise ratio.
COMBINE_SIGNAL = True
# Unit on time axis (used in spectrograms); 
# should be "seconds" or "minutes".
TIME_UNITS = "minutes"
# Unit on the frequency axis (used in FFT and spectrograms); 
# should be "Hz" or "cpm".
FREQ_UNIT = "cpm"
# Decomposition method. Either "ICA" or "PCA", or None for no decomposition.
DECOMP_METHOD = None

# EXPERIMENT
# Duration of the baseline in seconds.
BASELINE_DUR = 0.75
# Duration of the post-stimulus duration (inter-trial interval) in seconds.
ITI_DUR = 0.75
# Duration of the stimulus in seconds.
STIM_DUR = 10.0
# Error margin (proportion) on the stimulus timing.
STIM_DUR_MARGIN = 0.6
# Trigger codes used for events in the experiment.
TRIGGERS = {}
TRIGGERS["fix_onset"] = 201
TRIGGERS["stim_onset"] = 202
TRIGGERS["stim_offset"] = 203
TRIGGERS["block_start"] = 210
TRIGGERS["block_end"] = 211
TRIGGERS["block_start_disgust"] = 212
TRIGGERS["block_end_disgust"] = 213
TRIGGERS["block_start_healthy"] = 214
TRIGGERS["block_end_healthy"] = 215
TRIGGERS["block_start_unhealthy"] = 216
TRIGGERS["block_end_unhealthy"] = 217
TRIGGERS["block_start_baseline"] = 218
TRIGGERS["block_end_baseline"] = 219
# Individual stimuli.
TRIGGERS["disgust"] = [  1,   2,   3,   4,   5,   6,   7,   8,   9,  10, \
     11,  12, 13, 14, 15]
TRIGGERS["healthy"] = [ 51,  52,  53,  54,  55,  56,  57,  58,  59,  60, \
     61,  62, 63, 63, 65]
TRIGGERS["unhealthy"] = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, \
    111, 112, 113, 114, 115]
TRIGGERS["baseline"] = []
# Number of conditions, and number of stimuli within each condition.
N_CONDITIONS = len(CONDITIONS)
N_STIMULI = 12
# Number of times each stimulus is repeated.
UNIQUE_TRIAL_REPEATS = 2
# Compute the expected duration of one condition (in samples, assuming 200 Hz).
BLOCK_DURATION = N_STIMULI * UNIQUE_TRIAL_REPEATS \
    * (BASELINE_DUR + STIM_DUR + ITI_DUR)
ASSUMED_SAMPLING_RATE = 200.0
N_FOR_FFT = round(BLOCK_DURATION * ASSUMED_SAMPLING_RATE)

# FILES AND FOLDERS
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data")
if DECOMP_METHOD is None:
    OUTDIR = os.path.join(DIR, "output_no-decomp")
else:
    OUTDIR = os.path.join(DIR, "output_{}".format(DECOMP_METHOD.lower()))
TMPDIR = os.path.join(OUTDIR, "all_reduced_data")
for dir_path in [OUTDIR, TMPDIR]:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

# PLOTTINGS
COL = {}
COL["disgust"] = "#8f5902"
COL["healthy"] = "#4e9a06"
COL["unhealthy"] = "#ce5c00"
COL["baseline"] = "#204a87"


# # # # #
# DATA PROCESSING

# Get all files in the data folder.
all_files = os.listdir(DATADIR)
all_files.sort()
# Count the files.
n_files = len(all_files)
# Count the number of files that are a TSV, and keep track of participant
# codes.
n_participants = 0
ppnames = []
for fi, fname in enumerate(all_files):
    fpath = os.path.join(DATADIR, fname)
    name, ext = os.path.splitext(os.path.basename(fpath))
    if ext == ".tsv":
        ppname = name.replace("_egg", "")
        if name not in EXCLUDED:
            n_participants += 1
            ppnames.append(ppname)
# Save all participant codes.
ppnames = numpy.array(ppnames)
pp_memmap = numpy.memmap(os.path.join(TMPDIR, "ppnames.dat"), \
    dtype="<U5", mode="w+", shape=ppnames.shape)
pp_memmap[:] = ppnames

# Compute the frequencies in the FFTs that we are going to run on all data.
f = scipy.fft.rfftfreq(N_FOR_FFT, 1.0 / ASSUMED_SAMPLING_RATE)
sel = (f >= HIGH_PASS) & (f <= LOW_PASS)
n_freqs = f[sel].shape[0]
# Save the frequencies in a temporary file
signal_freq = numpy.memmap(os.path.join(TMPDIR, "f.dat"), \
    dtype=numpy.float64, mode="w+", shape=n_freqs)
signal_freq[:] = f[sel]
# Create a temporary file to hold average or peak gastric data in.
shape = (n_participants, N_CONDITIONS, n_freqs)
signal_power_shape = numpy.memmap(os.path.join(TMPDIR, "p_shape.dat"), \
    dtype=numpy.int64, mode="w+", shape=len(shape))
signal_power_shape[:] = shape
signal_power = numpy.memmap(os.path.join(TMPDIR, "p.dat"), \
    dtype=numpy.float64, mode="w+", shape=shape)
signal_power[:] = numpy.NaN
# Create a temporary file to hold gastric channel data in. FOR EACH CHANNEL
# SEPARATELY (following pre-registered analysis).
shape = (n_participants, 4, N_CONDITIONS, n_freqs)
signal_power_ch_shape = numpy.memmap(os.path.join(TMPDIR, "p_ch_shape.dat"), \
    dtype=numpy.int64, mode="w+", shape=len(shape))
signal_power_ch_shape[:] = shape
signal_power_ch = numpy.memmap(os.path.join(TMPDIR, "p_ch.dat"), \
    dtype=numpy.float64, mode="w+", shape=shape)
signal_power_ch[:] = numpy.NaN
# Create a temporary file to hold only the gastric peak in. This isn't really
# used. I thought it would be helpful, but it's easier just to grab the same
# data from the saved FFTs above.
shape = (n_participants, N_CONDITIONS)
gastric_peak_shape = numpy.memmap(os.path.join(TMPDIR, \
    "gastric_peak_shape.dat"), dtype=numpy.int64, mode="w+", shape=len(shape))
gastric_peak_shape[:] = shape
gastric_peak = numpy.memmap(os.path.join(TMPDIR, "gastric_peak.dat"), \
    dtype=numpy.float64, mode="w+", shape=shape)
gastric_peak[:] = numpy.NaN

# Run through all files.
for ppi, ppname in enumerate(ppnames):
    
    fname = "{}_egg.tsv".format(ppname)
    fpath = os.path.join(DATADIR, fname)
    name, ext = os.path.splitext(os.path.basename(fpath))
    
    print("Processing file '{}'".format(fname))
    print("\tThis is participant {}/{}".format(ppi+1, n_participants))
    
    if ppname in EXCLUDED:
        print("\tThis participant is excluded; skipping processing.")
        continue
    
    if not os.path.isfile(fpath):
        raise Exception("Could not find file at path '{}'".format(fpath))
    
    individual_output_dir = os.path.join(OUTDIR, ppname)
    if not os.path.isdir(individual_output_dir):
        os.mkdir(individual_output_dir)
    
    # Load experimental procedure data.
    exp_fname = os.path.join("{}.txt".format(ppname))
    exp_fpath = os.path.join(DATADIR, exp_fname)
    with open(exp_fpath, "r") as f:
        header = f.readline()
    header = header.replace("\n", "").split("\t")
    raw = numpy.loadtxt(exp_fpath, dtype=str, delimiter="\t", skiprows=1)
    exp_info = {}
    for i, var in enumerate(header):
        exp_info[var] = raw[:,i]
        if var in ["trialnr", "stim_nr", "stim_trigger"]:
            exp_info[var] = exp_info[var].astype(numpy.int64)
        elif var in ["fix_onset", "stim_onset", "stim_offset"]:
            exp_info[var] = exp_info[var].astype(numpy.float64)
    # Find when the blocks started.
    block_onset = numpy.hstack([0, numpy.where( \
        exp_info["img_category"][:-1] != exp_info["img_category"][1:])[0]+1])
    t0 = exp_info["stim_onset"][0]
    for condition in numpy.unique(exp_info["img_category"]):
        exp_info["block_start_{}".format(condition)] = []
        exp_info["block_end_{}".format(condition)] = []
    for i, onset in enumerate(block_onset):
        con = exp_info["img_category"][onset]
        exp_info["block_start_{}".format(con)].append( \
            exp_info["stim_onset"][onset] - t0)
        if i == len(block_onset) - 1:
            offset = -1
        else:
            offset = block_onset[i+1] - 1
        exp_info["block_end_{}".format(con)].append( \
            exp_info["stim_offset"][offset] - t0)

    # Load EGG data.
    print("\tLoading data from file...")
    raw = read_file(fpath)
    print("\tLoaded data with shape {}".format(raw.shape))
    
    # Extract the physiological data.
    print("\tExtracting physiology...")
    data, t, triggers, sampling_rate = get_physiology(raw, BOARD)
    
    # Create a figure for visual inspection of the whole data range.
    n_rows = data.shape[0]
    n_cols = 6
    fig, axes = pyplot.subplots(figsize=(n_cols*15.0,n_rows*5.0), dpi=100.0, \
        nrows=n_rows, ncols=n_cols)
    # Plot the raw data.
    axes[0,0].set_title("Raw signal", fontsize=32)
    for channel in range(data.shape[0]):
        axes[channel,0].plot(data[channel,:], "-", lw=1)
        axes[channel,0].set_ylabel("Channel {}".format(channel+1) + \
            r" ($\mu$V)", fontsize=16)
    
    # Mean-subtract.
    print("\tApplying mean subtraction...")
    axes[0,1].set_title("Centred signal", fontsize=32)
    for channel in range(data.shape[0]):
        channel_mean = numpy.nanmean(data[channel,:])
        data[channel,:] -= channel_mean
        print("Channel mean was {}, is now {}".format(channel_mean, \
            numpy.nanmean(data[channel,:])))
        axes[channel,1].plot(data[channel,:], "-", lw=1)

    # Exclude high-amplitude sections.
    if OUTLIER_T is not None:
        print("\tMedian-replacing outliers...")
        axes[0,2].set_title("High-amplitude rejection", fontsize=32)
        t0 = time.time()
        med = numpy.nanmedian(data, axis=1)
        d = numpy.abs(data - med.reshape(-1,1))
        d_med = numpy.median(d, axis=1)
        sd = 1.4826 * d_med
        threshold = OUTLIER_T * sd
        for channel in range(data.shape[0]):
            replace = d[channel,:] > threshold[channel]
            data[channel,replace] = med[channel]
        t1 = time.time()
        print("\t\tCompleted in {:.3f} seconds!".format(t1-t0))
        for channel in range(data.shape[0]):
            axes[channel,2].plot(data[channel,:], "-", lw=1)

    # Filter target frequencies.
    print("\tApplying Butterworth filter...")
    axes[0,3].set_title("Filtered signal", fontsize=32)
    for channel in range(data.shape[0]):
        data[channel,:] = butter_bandpass_filter(data[channel,:], \
            HIGH_PASS, LOW_PASS, sampling_rate)
        data[channel,:] -= numpy.mean(data[channel,:])
        axes[channel,3].plot(data[channel,:], "-", lw=1)
    
    # Employ movement filter.
    if MOVEMENT_FILTER:
        print("\tFiltering suspected movement out of the signal...")
        axes[0,4].set_title("Movement filter", fontsize=32)
        data_pre_filter = data.copy()
        t0 = time.time()
        data, noise = movement_filter(data, sampling_rate, \
            freq=FREQ_OF_INTEREST, window=1.0)
        data_post_filter = data.copy()
        t1 = time.time()
        print("\t\tCompleted in {:.3f} seconds!".format(t1-t0))
        for channel in range(data.shape[0]):
            axes[channel,4].plot(data[channel,:], "-", lw=1)

    # Decompose into ICA signal.
    ica_denoising_failed = False
    if SNR_THRESHOLD is not None:
        print("\tApplying ICA denoising...")
        # Create a new figure to plot ICA components in.
        n_rows = data.shape[0]
        n_cols = 1
        fig_, axes_ = pyplot.subplots(figsize=(n_cols*8.0,n_rows*5.0), \
            dpi=100.0, nrows=n_rows, ncols=n_cols)
        # Fit the ICA model to the data (transposed to fit the required input 
        # shape).
        ica = FastICA(random_state=19)
        ica_components = ica.fit_transform(numpy.copy(data).T).T
        # Go through all channels.
        n_filtered_channels = 0
        for channel in range(ica_components.shape[0]):
            # Compute the signal magnitude for this component.
            f, p = compute_signal_magnitude(numpy.copy( \
                ica_components[channel,:]), ica_components.shape[1], \
                sampling_rate, HIGH_PASS, LOW_PASS)
            # Compute peak power in the normogastric range. This is our main 
            # signal of interest.
            sel = (f >= BANDS["normo"][0]) & (f <= BANDS["normo"][1])
            p_signal = numpy.nanmax(p[sel])
            # Compute the "noise" (mean power in other ranges).
            p_noise = numpy.nanmean(p[numpy.invert(sel)])
            # Zero-out this component if the signal-noise ratios is too low.
            snr = p_signal / p_noise
            excluded = snr < SNR_THRESHOLD
            print("\t\tComponent {}: SNR={} ({})".format(channel, snr, \
                ["included", "excluded"][int(excluded)]))
            if excluded:
                ica_components[channel,:] = 0.0
                n_filtered_channels += 1
            # Plot the frequencies.
            axes_[channel].plot(f, p, "-")
            axes_[channel].set_xlabel("Frequency ({})".format(FREQ_UNIT), \
                fontsize=14)
            axes_[channel].set_ylabel(r"Signal magnitude ($\mu$V)", \
                fontsize=14)
            axes_[channel].set_title("Component {}: SNR={} ({})".format( \
                channel, round(snr,5), \
                ["included", "excluded"][int(excluded)]))
            if FREQ_UNIT == "cpm":
                xticklabels = numpy.round(axes_[channel].get_xticks()*60.0, 1)
                axes_[channel].set_xticklabels(xticklabels)
        # Save and close ICA component figure.
        fig_.savefig(os.path.join(individual_output_dir, \
            "denoising_ICA_components.png"))
        pyplot.close(fig_)
        # Stop if we have no data left to inverse transform.
        if n_filtered_channels < data.shape[0]:
            # Recombine the components into the original signal space now that
            # the non-gastric channels have been filtered out.
            data = ica.inverse_transform(ica_components.T).T
            # Ensure the array is C-contiguous, otherwise some of our data
            # functions can't deal with it. (This is specific to the 
            # Cythonised functions, which assume C-contiguous arrays for 
            # improved efficiency.)
            data = numpy.ascontiguousarray(data)
        else:
            ica_denoising_failed = True
        # Plot the resulting signal.
        axes[0,5].set_title("ICA de-noising", fontsize=32)
        for channel in range(data.shape[0]):
            axes[channel,5].plot(data[channel,:], "-", lw=1)

    # Save and close the inspection figure.
    fig.savefig(os.path.join(individual_output_dir, "whole_signal.png"))
    pyplot.close(fig)
    
    # If not enough signal was found for this participant, skip to the next.
    if ica_denoising_failed:
        print("\tNo gastric signal found; skipping to next participant")
        continue

    # Plot signal, estimated movement noise, and filtered signal.
    if MOVEMENT_FILTER:
        print("\tPlotting movement noise and filtered signal spectrograms...")
        mode = "magnitude"
        interpolation = "gaussian"
        signals = [data_pre_filter, noise, data_post_filter]
        n_cols = len(signals)
        fig, axes = pyplot.subplots(figsize=(10.0*n_cols,4*6.0), \
            dpi=100, ncols=n_cols, nrows=4)
        segment_secs = SPECTROGRAM_SEGMENT_DURATION_SECS
        
        for col, signal_ in enumerate(signals):
            freqs, times, power = spectrogram(signal_, sampling_rate, \
                segment_secs, mode=mode)

            for channel in range(signal_.shape[0]):
                rough_amp = signal_[channel,:].max() \
                    - signal_[channel,:].min()
                vmax = {"magnitude":0.03*rough_amp, \
                    "psd":0.1*rough_amp}[mode]
                vmax = {"magnitude":None, "psd":None}[mode]
                ax = axes[channel,col]
                ax.set_title("Channel {}".format(channel+1), \
                    loc="left", fontsize=16)
                sel = (freqs >= HIGH_PASS) & (freqs <= LOW_PASS)
                ax.imshow(power[channel,sel,:], aspect="auto", \
                    origin="lower", interpolation=interpolation, \
                    vmax=vmax)

                ax.set_xlabel("Time ({})".format(TIME_UNITS), \
                    fontsize=16)
                xstep = times.shape[0] // 10
                xticks = list(range(0, times.shape[0], xstep))
                if TIME_UNITS == "seconds":
                    xticklabels = numpy.round(times[xticks], 1)
                elif TIME_UNITS == "minutes":
                    xticklabels = numpy.round(times[xticks] / 60, 1)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, fontsize=12)

                ax.set_ylabel("Frequency ({})".format(FREQ_UNIT), \
                    fontsize=16)
                ystep = max(1, freqs[sel].shape[0] // 10)
                yticks = list(range(0, freqs[sel].shape[0], ystep))
                if FREQ_UNIT == "Hz":
                    yticklabels = numpy.round(freqs[sel][yticks], 1)
                elif FREQ_UNIT == "cpm":
                    yticklabels = numpy.round( \
                        freqs[sel][yticks] * 60, 1)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels, fontsize=12)

        fig.savefig(os.path.join(individual_output_dir, \
            "movement_channel_spectrogram_{}_interp-{}.png".format(mode, \
                interpolation)))
        pyplot.close(fig)
    
    # Create a figure with signal and FFT for each channel.
    print("\tPlotting waves and fast Fournier transform...")
    n_rows = data.shape[0]
    n_cols = 2
    fig, axes = pyplot.subplots(figsize=(n_cols*8.0,n_rows*5.0), dpi=100.0, \
        nrows=n_rows, ncols=n_cols)
    for row in range(n_rows):
        channel = row
        # Take a slice of several minutes from the middle of the recording.
        slice_len = 3 * 60 * sampling_rate
        si = data.shape[1] // 2 - slice_len // 2
        ei = si + slice_len
        # Plot only the slice, so we can actually see what is going on.
        axes[row,0].plot(t[si:ei]-t[0], data[channel,si:ei], "-")
        axes[row,0].set_title("Channel {}".format(channel+1), loc="left", \
            fontsize=16)
        axes[row,0].set_xlabel("Time (samples)", fontsize=14)
        axes[row,0].set_ylabel(r"Amplitude ($\mu$V)", fontsize=14)
        # Compute the signal magnitude in frequency space.
        f, p = compute_signal_magnitude(numpy.copy(data[channel,:]), \
            N_FOR_FFT, sampling_rate, HIGH_PASS, LOW_PASS)
        # Plot the frequencies.
        axes[row,1].plot(f, p, "-")
        axes[row,1].set_xlabel("Frequency ({})".format(FREQ_UNIT), fontsize=14)
        axes[row,1].set_ylabel(r"Signal magnitude ($\mu$V)", fontsize=14)
        if FREQ_UNIT == "cpm":
            xticklabels = numpy.round(axes[row,1].get_xticks() * 60.0, 1)
            axes[row,1].set_xticklabels(xticklabels)
    # Save and close figure.
    fig.savefig(os.path.join(individual_output_dir, \
        "channel_waves_and_power.png"))
    pyplot.close(fig)

    # Compute and plot several spectrograms.
    print("\tPlotting spectrogram...")
    mode = "magnitude"
    interpolation = "gaussian"
    fig, axes = pyplot.subplots(figsize=(10.0,4*6.0), dpi=100, nrows=4)
    segment_secs = SPECTROGRAM_SEGMENT_DURATION_SECS
    freqs, times, power = spectrogram(data, sampling_rate, \
        segment_secs, mode=mode)

    for channel in range(data.shape[0]):
        rough_amp = data[channel,:].max() - data[channel,:].min()
        vmax = {"magnitude":0.03*rough_amp, "psd":0.1*rough_amp}[mode]
        ax = axes[channel]
        ax.set_title("Channel {}".format(channel+1), loc="left", \
            fontsize=16)
        sel = (freqs >= HIGH_PASS) & (freqs <= LOW_PASS)
        ax.imshow(power[channel,sel,:], aspect="auto", \
            origin="lower", interpolation=interpolation, \
            vmax=vmax)

        ax.set_xlabel("Time ({})".format(TIME_UNITS), fontsize=16)
        xstep = times.shape[0] // 10
        xticks = list(range(0, times.shape[0], xstep))
        if TIME_UNITS == "seconds":
            xticklabels = numpy.round(times[xticks], 1)
        elif TIME_UNITS == "minutes":
            xticklabels = numpy.round(times[xticks] / 60, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=12)

        ax.set_ylabel("Frequency ({})".format(FREQ_UNIT), fontsize=16)
        ystep = max(1, freqs[sel].shape[0] // 10)
        yticks = list(range(0, freqs[sel].shape[0], ystep))
        if FREQ_UNIT == "Hz":
            yticklabels = numpy.round(freqs[sel][yticks], 1)
        elif FREQ_UNIT == "cpm":
            yticklabels = numpy.round(freqs[sel][yticks] * 60, 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=12)

    fig.savefig(os.path.join(individual_output_dir, \
        "channel_spectrogram_{}_interp-{}.png".format(mode, \
            interpolation)))
    pyplot.close(fig)

    # Independent component analysis.
    print("\tPerforming {} decomposition...".format(DECOMP_METHOD))
    # If no decomposition is to be done, simply copy the channels as they are.
    if DECOMP_METHOD is None:
        data_ica = numpy.copy(data)
    # Perform an ICA on the physiological data. This is a common approach to 
    # decompose EEG signal, so might well work here too.
    elif DECOMP_METHOD == "ICA":
        # Fit the ICA model to the data (transposed to fit the required input 
        # shape).
        ica = FastICA(random_state=19)
        s = ica.fit_transform(data.T)
        # This is how to "unmix" the data with all components.
        #data_estimated = (numpy.dot(s, ica.mixing_.T) + ica.mean_).T
        # Unmix the data one component at a time.
        data_ica = numpy.zeros(data.shape, data.dtype)
        for channel in range(data_ica.shape[0]):
            #data_ica[channel,:] = numpy.dot(s, ica.mixing_.T) + ica.mean_
            include = numpy.zeros(data_ica.shape[0], dtype=bool)
            include[channel] = True
            data_ica[channel,:] = (numpy.dot(s[:,include], \
                ica.mixing_[:,include][include,:].T) + ica.mean_[include]).T
    # Decomp with PCA? Not the traditional method in EEG, but implemented here 
    # anyways.
    elif DECOMP_METHOD == "PCA":
        ica = PCA(random_state=19)
        data_ica = ica.fit_transform(data.T).T

    # Compute the gastric power ratio in each component.
    gastric_power_ratio = numpy.zeros(data_ica.shape[0], dtype=numpy.float64)
    
    # Create a new figure for component plotting.
    n_cols = 2
    n_rows = data_ica.shape[0]
    fig, axes = pyplot.subplots(figsize=(n_cols*8.0,n_rows*5.0), \
        dpi=100.0, nrows=n_rows, ncols=n_cols)
    for channel in range(data_ica.shape[0]):
        # Compute the signal magnitude in frequency space.
        f, p = compute_signal_magnitude(numpy.copy(data_ica[channel,:]), \
            N_FOR_FFT, sampling_rate, HIGH_PASS, LOW_PASS)
        # Select the target frequencies.
        sel_target = numpy.zeros(f.shape, dtype=bool)
        for ica_target_freq in ICA_TARGET_FREQ:
            st = (f >= ica_target_freq[0]) & (f <= ica_target_freq[1])
            sel_target[st] = True
        sel_background = numpy.invert(sel_target)
        # Compute target:background power ratio.
        gastric_power_ratio[channel] = numpy.max(p[sel_target]) \
            / numpy.mean(p[sel_background])

        # Plot example data from the component.
        row = channel
        axes[row,0].plot(t[si:ei]-t[0], data_ica[channel,si:ei], "-")
        tit = "Component {}, power-ratio={}".format(channel+1, \
            numpy.round(gastric_power_ratio[channel],3))
        axes[row,0].set_title(tit, loc="left", fontsize=16)
        axes[row,0].set_xlabel("Time (samples)", fontsize=14)
        axes[row,0].set_ylabel(r"Amplitude ($\mu$V)", fontsize=14)
        # Plot the frequency spectrum.
        axes[row,1].plot(f, p, "-")
        axes[row,1].plot(f[sel_target], p[sel_target], "-", color="#FF69B4")
        axes[row,1].set_xlabel("Frequency ({})".format(FREQ_UNIT), fontsize=14)
        axes[row,1].set_ylabel(r"Signal magnitude ($\mu$V)", fontsize=14)

        if FREQ_UNIT == "cpm":
            #xticklabels = [txt.get_position()[0] for txt in \
            #    axes[row,1].get_xticklabels()]
            #xticklabels = [str(float(lbl)*60) for lbl in xticklabels]
            xticklabels = numpy.round(axes[row,1].get_xticks() * 60.0, 1)
            axes[row,1].set_xticklabels(xticklabels)

    # Save and close figure.
    fig.savefig(os.path.join(individual_output_dir, \
        "component_waves_and_power.png"))
    pyplot.close(fig)

    # Choose the gastric component (== highest power ratio).
    gastric_component = numpy.argmax(gastric_power_ratio)

    # PCA directions are arbitrary, so we'll flip them so that the gastric
    # channel is aligned with the other channels (by computing correlations,
    # and flipping the PCA if the average correlation between gastric 
    # component and channels is negative.)
    if DECOMP_METHOD == "PCA":
        r = numpy.zeros(data.shape[0], dtype=numpy.float64)
        for channel in range(data.shape[0]):
            r[channel], p_val = scipy.stats.pearsonr(data[channel,:], \
                data_ica[gastric_component,:])
        print("\tCorrelations between gastric component and channels:")
        print("\t\t{}".format(numpy.round(r,2)))
        if numpy.mean(r) < 0:
            print("\tFlipping components (mean R={})".format( \
                round(numpy.mean(r),2)))
            data_ica *= -1

    # Compute the spectrograms of the EGG component.
    mode = "magnitude"
    interpolation = "gaussian"
    fig, ax = pyplot.subplots(figsize=(10.0,6.0), dpi=100)
    segment_secs = SPECTROGRAM_SEGMENT_DURATION_SECS
    freqs, times, power = spectrogram(data_ica[gastric_component], \
        sampling_rate, segment_secs, mode=mode)
    vmax=None
    ax.set_title("Gastric component", loc="left", fontsize=16)
    sel = (freqs >= HIGH_PASS) & (freqs <= LOW_PASS)
    ax.imshow(power[sel,:], aspect="auto", \
        origin="lower", interpolation=interpolation, \
        vmax=vmax)
    ax.set_xlabel("Time ({})".format(TIME_UNITS), fontsize=16)
    xstep = times.shape[0] // 10
    xticks = list(range(0, times.shape[0], xstep))
    if TIME_UNITS == "seconds":
        xticklabels = numpy.round(times[xticks], 1)
    elif TIME_UNITS == "minutes":
        xticklabels = numpy.round(times[xticks] / 60, 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.set_ylabel("Frequency ({})".format(FREQ_UNIT), fontsize=16)
    ystep = max(1, freqs[sel].shape[0] // 10)
    yticks = list(range(0, freqs[sel].shape[0], ystep))
    if FREQ_UNIT == "Hz":
        yticklabels = numpy.round(freqs[sel][yticks], 1)
    elif FREQ_UNIT == "cpm":
        yticklabels = numpy.round(freqs[sel][yticks] * 60, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=12)
    fig.savefig(os.path.join(individual_output_dir, \
        "component_spectrogram_{}_interp-{}.png".format(mode, \
            interpolation)))
    pyplot.close(fig)

    
    # # # # #
    # EXPERIMENT DATA
    
    print("\tExtracting triggers.")
    
    # Convert triggers to integers.
    triggers = triggers.astype(numpy.int32)
    
    # Find image presentation windows by comparing stimulus onsets and offsets.
    # First, find all onset and offset triggers.
    triggered_onset = numpy.where(triggers==TRIGGERS["stim_onset"])[0]
    triggered_offset = numpy.where(triggers==TRIGGERS["stim_offset"])[0]
    
    # Compute the minimum and maximum onset-offset distances.
    i_stim_dur_min = int(numpy.floor((STIM_DUR * (1.0 - STIM_DUR_MARGIN)) \
        / (1.0 / sampling_rate)))
    i_stim_dur_max = int(numpy.ceil((STIM_DUR * (1.0 + STIM_DUR_MARGIN)) \
        / (1.0 / sampling_rate)))
    i_stim_dur = round(STIM_DUR / (1.0 / sampling_rate))
    
    # Go through all marked onsets and offsets to ensure we're imputing
    # missing triggers on the basis of the expected stimulus duration.
    onset = []
    offset = []
    for i in range(triggered_onset.shape[0]):
        # Find the closest offset to where we're expecting to find an offset 
        # (i.e. onset plus stimulus duration). This should be at the same 
        # index in the trigger arrays, but not if an offset went missing!
        i_onset = triggered_onset[i]
        i_ = numpy.argmin(numpy.abs(triggered_offset - (i_onset+i_stim_dur)))
        i_offset = triggered_offset[i_]
        # Accept the match as One And Only True Match if the offset is within
        # the margins.
        accepted = (i_offset > i_onset + i_stim_dur_min) \
            & (i_offset < i_onset + i_stim_dur_max)
        # Create a new offset if we were missing one (make sure it's within
        # the data, with a margin of 1 extra, to add and end-of-block behind
        # it if necessary).
        if not accepted:
            i_offset = min(i_onset + i_stim_dur, triggers.shape[0]-2)
            print("\t\tMissing offset for onset at index {}".format(i))
        onset.append(i_onset)
        offset.append(i_offset)

    for i in range(triggered_offset.shape[0]):
        # Find the closest onset to where we're expecting to find an onset 
        # (i.e. offset minus stimulus duration). This should be at the same 
        # index in the trigger arrays, but not if an onset went missing!
        i_offset = triggered_offset[i]
        i_ = numpy.argmin(numpy.abs(triggered_onset - (i_offset-i_stim_dur)))
        i_onset = triggered_onset[i_]
        # Skip if the onset was already captured.
        if i_offset in offset:
            continue
        # Accept the match as One And Only True Match if the offset is within
        # the margins.
        accepted = (i_onset < i_offset - i_stim_dur_min) \
            & (i_onset > i_offset - i_stim_dur_max)
        # Create a new onset if we were missing one (make sure it's within
        # the data, with a margin of 1, to add a start-of-block in front if
        # necessary).
        if not accepted:
            i_onset = max(i_offset-i_stim_dur, 1)
            print("\t\tMissing onset for offset at index {}".format(i))
        onset.append(i_onset)
        offset.append(i_offset)
    
    onset = numpy.array(onset)
    offset = numpy.array(offset)
    
    # Mark stretches of image presentation, and find minimum and maximum 
    # durations (in samples).
    sel_img = numpy.zeros(triggers.shape, dtype=bool)
    min_len = numpy.inf
    max_len = 0
    for i in range(onset.shape[0]):
        sel_img[onset[i]:offset[i]] = True
        if offset[i] - onset[i] < min_len:
            min_len = offset[i] - onset[i]
        if offset[i] - onset[i] > max_len:
            max_len = offset[i] - onset[i]
    
    # Find which image presentations were in which condition.
    stim = numpy.where((triggers>0) & (triggers<200))[0]
    trial = {}
    for condition in CONDITIONS:
        trial[condition] = numpy.zeros(stim.shape[0], dtype=bool)
        for i, trig in enumerate(triggers[stim]):
            if trig in TRIGGERS[condition]:
                trial[condition][i] = True
    
    # Identify image presentation windows.
    img_win = {}
    for condition in trial.keys():
        img_win[condition] = numpy.zeros(triggers.shape, dtype=bool)
        for i in range(onset[trial[condition]].shape[0]):
            img_win[condition] \
                [onset[trial[condition]][i]:offset[trial[condition]][i]] = True

    # In very rare cases, triggers are dropped. This bit of code is to add the 
    # block start or end triggers by adding them before the first (start) or 
    # after the last (end) stimulus in this condition.
    # Note that sometimes entire conditions aren't run if the experiment was
    # prematurely ended. In this case, the back-up option will not work.
    complete_dataset = True
    for condition in CONDITIONS:
        for moment in ["start", "end"]:
            # Construct the trigger name.
            trigger_type = "block_{}_{}".format(moment, condition)
            # Check if the trigger is missing.
            if numpy.sum(triggers==TRIGGERS[trigger_type]) == 0:
                # Get the condition name.
                _, moment, condition = trigger_type.split("_")
                # Check if there are any stimuli within this condition.
                if numpy.sum(trial[condition]) > 0:
                    if moment == "start":
                        i = numpy.min(onset[trial[condition]]) - 1
                    elif moment == "end":
                        i = numpy.max(offset[trial[condition]]) + 1
                    triggers[i] = TRIGGERS[trigger_type]
                else:
                    complete_dataset = False
    
    # Skip further processing if the dataset is missing a condition.
    if not complete_dataset:
        print("\tNot all conditions present in data; no further processing.")
        continue
    
    # Find block onsets and offsets.
    block_onset = {}
    block_offset = {}
    for condition in CONDITIONS:
        block_onset[condition] = numpy.where( \
            triggers==TRIGGERS["block_start_{}".format(condition)])[0][0]
        block_offset[condition] = numpy.where( \
            triggers==TRIGGERS["block_end_{}".format(condition)])[0][0]
    sel_per_cond = {}
    for condition in CONDITIONS:
        sel_per_cond[condition] = numpy.zeros(triggers.shape, dtype=bool)
        sel_per_cond[condition] \
            [block_onset[condition]:block_offset[condition]] = True
    
    # Plot raw signal, coloured by condition.
    n_rows = data.shape[0]
    n_cols = 2
    fig, axes = pyplot.subplots(figsize=(n_cols*8.0,n_rows*6.0), dpi=100, \
        nrows=n_rows, ncols=n_cols)
    for col, d in enumerate([data, data_ica]):
        for row in range(n_rows):
            # Skip if there are fewer ICA components than data channels.
            if row >= d.shape[0]:
                continue
            signal_ = d[row,:]
            ax = axes[row,col]
            if col == 0:
                tit = "Channel"
            elif col == 1:
                tit = "Component"
            ax.set_title("{} {}".format(tit, row+1), \
                fontsize=16)
            # Plot all data.
            ax.plot(t, signal_, "-", lw=1, color="#000000")
            # Plot the windows.
            for condition in img_win.keys():
                colour = COL[condition]
                ax.plot(t[img_win[condition]], signal_[img_win[condition]], \
                    "-", lw=1, color=colour)
    fig.savefig(os.path.join(individual_output_dir, \
        "exp_signal-per-condition.png"))
    pyplot.close(fig)

    # Compute signal power per channel and per component.
    shape = (data.shape[0], len(CONDITIONS), signal_freq.shape[0])
    p_channel = numpy.zeros(shape, dtype=numpy.float64) * numpy.nan
    p_component = numpy.zeros(shape, dtype=numpy.float64) * numpy.nan
    for di, d in enumerate([data, data_ica]):
        for ci, condition in enumerate(CONDITIONS):
            sel = sel_per_cond[condition]
            for channel in range(d.shape[0]):
                # Compute the signal magnitude in frequency space.
                if di == 0:
                    p = p_channel
                elif di == 1:
                    p = p_component
                f, p[channel, ci, :] = compute_signal_magnitude( \
                    numpy.copy(d[channel,sel]), N_FOR_FFT, sampling_rate, \
                    HIGH_PASS, LOW_PASS)

    # Average the FFTs into the signal power.
    if COMBINE_SIGNAL:
        for ci, condition in enumerate(CONDITIONS):
            signal_power[ppi,ci,:] = numpy.nanmean(p_channel[:,ci,:], axis=0)
    # Only take the signal from the component with the strongest gastric
    # signature (i.e. highest normogastric signal-noise ratio).
    else:
        for ci, condition in enumerate(CONDITIONS):
            signal_power[ppi,ci,:] = p_component[gastric_component,ci,:]
    
    # Store power for each channel independently.
    signal_power_ch[ppi,:,:,:] = p[:,:,:]
    
    # Compute the peak signal in the normogastric range.
    sel_target = (signal_freq >= BANDS["normo"][0]) \
        & (signal_freq <= BANDS["normo"][1])
    for ci, condition in enumerate(CONDITIONS):
        gastric_peak[ppi,ci] = numpy.nanmax(signal_power[ppi,ci,sel_target])
    
    # Plot FFTs for each channel, component, and condition.
    print("\tPlotting fast Fournier transform for experimental conditions...")
    n_rows = data.shape[0]
    n_cols = 2
    fig, axes = pyplot.subplots(figsize=(n_cols*8.0,n_rows*5.0), dpi=100.0, \
        nrows=n_rows, ncols=n_cols)
    axes[0,0].set_title("Channels", fontsize=24)
    axes[0,1].set_title("Components", fontsize=24)
    for col, p in enumerate([p_channel, p_component]):
        for row in range(n_rows):
            channel = row
            for ci, condition in enumerate(CONDITIONS):
                sel = sel_per_cond[condition]
                colour = COL[condition]
                # Plot the frequencies.
                axes[row,col].plot(signal_freq, p[channel,ci,:], "-", \
                    lw=3, color=colour, alpha=0.5, label=condition)
                axes[row,col].set_xlabel("Frequency ({})".format(FREQ_UNIT), \
                    fontsize=14)
                axes[row,col].set_ylabel(r"Signal magnitude ($\mu$V)", \
                    fontsize=14)
                if FREQ_UNIT == "cpm":
                    xticklabels = numpy.round( \
                        axes[row,col].get_xticks() * 60.0, 1)
                    axes[row,col].set_xticklabels(xticklabels)
    # Save and close figure.
    axes[-1,-1].legend(loc="upper right", fontsize=10)
    fig.savefig(os.path.join(individual_output_dir, \
        "exp_power-per-condition.png"))
    pyplot.close(fig)
    
    # Compute the peak power in the normogastric range, and plot a single 
    # figure with the signal power per condition for this participant.
    if COMBINE_SIGNAL:
        fig, ax = pyplot.subplots()
        for ci, condition in enumerate(CONDITIONS):
            # Choose the line colour.
            colour = COL[condition]
            # Plot the frequencies.
            ax.plot(signal_freq, signal_power[ppi,ci,:], "-", lw=3, \
                color=colour, alpha=0.5, label=condition.capitalize())
            ax.set_xlabel("Frequency ({})".format(FREQ_UNIT), fontsize=14)
            ax.set_ylabel(r"Signal magnitude ($\mu$V)", fontsize=14)
            if FREQ_UNIT == "cpm":
                xticklabels = numpy.round(ax.get_xticks() * 60.0, 1)
                ax.set_xticklabels(xticklabels)
            ax.legend(loc="upper right", fontsize=14)
        # Save and close figure.
        fig.savefig(os.path.join(individual_output_dir, \
            "avg_signal_power_per_condition.png"))
        pyplot.close(fig)

