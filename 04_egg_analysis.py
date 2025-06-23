#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy

import numpy
import scipy.stats
from matplotlib import pyplot

import pandas
from statsmodels.api import MixedLM
from statsmodels.tools.eval_measures import aic, bic


# # # # #
# CONSTANTS

EXCLUDED = [ \
    # Participant 123 had an almost empty EGG file, likely due to device 
    # malfunction.
    "pp123", 
    # Crash during baseline data collection. The data is in pp139.
    "pp139baseline", \
    # Pilot participants.
    "pp201", \
    "pp202", \
    # Participant participated twice, second time under pp334. (Somehow they
    # managed to sign up twice.)
    "pp334", \
    ]

# Included conditions.
CONDITIONS = ["disgust", "healthy", "unhealthy", "baseline"]

# SIGNAL
# Unit on time axis (used in spectrograms); 
# should be "seconds" or "minutes".
TIME_UNITS = "minutes"
# Unit on the frequency axis (used in FFT and spectrograms); 
# should be "Hz" or "cpm".
FREQ_UNIT = "cpm"
# Decomposition method. Either "ICA" or "PCA", or None for no decomposition.
# If None, the channel with the highest gastric ratio (i.e. max in target 
# range compared to mean of all other signal) will be chosen to derive peak 
# data from. (Note that none of this is actually important for our analysis,
# which uses data from all four channels.)
DECOMP_METHOD = None

# FILES AND FOLDERS
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data")
DEMFILE = os.path.join(DATADIR, "demographics.csv")
QFILE = os.path.join(DATADIR, "data_questionnaires.csv")
RFILE = os.path.join(DATADIR, "data_ratings.csv")
RSFILE = os.path.join(DATADIR, "data_ratings_per_stimulus.csv")
HFILE = os.path.join(DATADIR, "data_heart_rate.csv")
if DECOMP_METHOD is None:
    OUTDIR = os.path.join(DIR, "output_no-decomp")
else:
    OUTDIR = os.path.join(DIR, "output_{}".format(DECOMP_METHOD.lower()))
TMPDIR = os.path.join(OUTDIR, "all_reduced_data")
SUMDIR = os.path.join(OUTDIR, "all_group_outcomes")
LMEDIR = os.path.join(SUMDIR, "lme_outcomes")
for dir_path in [SUMDIR, LMEDIR]:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

# Variables in the questionnaires, and their variable names in the file's
# header.
QUESTIONNAIRE_DICT = { \
    "ton":                  "TON_SUM", \
    "tos_healthy":          "TOS_HO_SUM", \
    "tos_nervosa":          "TOS_ON_SUM", \
    "sticsa_somatic":       "STICSA_SOM_SUM", \
    "sticsa_cognitive":     "STICSA_COG_SUM", \
    }

# Plot colours.
COL = {}
COL["disgust"] = "#8f5902"
COL["healthy"] = "#4e9a06"
COL["unhealthy"] = "#c4a000"
COL["baseline"] = "#204a87"


# # # # #
# LOAD DATA

# Load participant codes.
pp_memmap = numpy.memmap(os.path.join(TMPDIR, "ppnames.dat"), dtype="<U5", \
    mode="r")
ppnames = numpy.copy(pp_memmap)
# Load the frequency range.
freq = numpy.memmap(os.path.join(TMPDIR, "f.dat"), dtype=numpy.float64, \
    mode="r")
freq_cpm = freq * 60
# Load signal magnitude data across the frequency range.
signal_power_ch_shape = numpy.memmap(os.path.join(TMPDIR, "p_ch_shape.dat"), \
    dtype=numpy.int64, mode="r")
signal_power_ch = numpy.memmap(os.path.join(TMPDIR, "p_ch.dat"), \
    dtype=numpy.float64, mode="r", shape=tuple(signal_power_ch_shape))
# Load the gastric peak data.
gastric_peak_shape = numpy.memmap(os.path.join(TMPDIR, \
    "gastric_peak_shape.dat"), dtype=numpy.int64, mode="r")
gastric_peak = numpy.memmap(os.path.join(TMPDIR, "gastric_peak.dat"), \
    dtype=numpy.float64, mode="r", shape=tuple(gastric_peak_shape))

# Load demographic data.
if os.path.isfile(DEMFILE):
    raw = numpy.loadtxt(DEMFILE, delimiter=",", unpack=True, dtype=str, \
        usecols=range(5))
    demographics = {}
    for i in range(raw.shape[0]):
        var = raw[i,0]
        val = raw[i,1:]
        try:
            val = val.astype(numpy.float64)
        except:
            pass
        demographics[var] = val
else:
    demographics = {"ppname": ppnames}

# Exclude demographic data for which we don't have a match.
dem = {}
pps = [ppname.lower() for ppname in demographics["ppnr"]]
for var in demographics.keys():
    if var == "ppnr":
        dem[var] = list(ppnames)
    dem[var] = numpy.zeros(len(ppnames), dtype=demographics[var].dtype)
    for ppi, ppname in enumerate(ppnames):
        if ppname.lower() in pps:
            i = pps.index(ppname.lower())
            dem[var][ppi] = demographics[var][i]
        else:
            try:
                dem[var][ppi] = numpy.NaN
            except:
                dem[var][ppi] = "NaN"

# Load questionnaire data.
if os.path.isfile(QFILE):
    raw = numpy.loadtxt(QFILE, delimiter=",", unpack=True, dtype=str)
    qdata = {}
    for i in range(raw.shape[0]):
        var = raw[i,0]
        val = raw[i,1:]
        try:
            val = val.astype(numpy.float64)
        except:
            pass
        qdata[var] = val
    # Sum scores.
    for var in QUESTIONNAIRE_DICT.keys():
        q_var = QUESTIONNAIRE_DICT[var]
        dem[var] = numpy.zeros(dem["ppnr"].shape, dtype=qdata[q_var].dtype)
        for i, ppname in enumerate(dem["ppnr"]):
            ppi = list(qdata["ppname"]).index(ppname)
            dem[var][i] = qdata[q_var][ppi]
    # Single items.
    for var in ["TON_{}".format(i) for i in range(1,18)] \
        + ["TOS_{}".format(i) for i in range(1,17)]:
        dem[var] = numpy.zeros(dem["ppnr"].shape, dtype=qdata[var].dtype)
        for i, ppname in enumerate(dem["ppnr"]):
            ppi = list(qdata["ppname"]).index(ppname)
            dem[var][i] = qdata[var][ppi]

# Load rating data.
if os.path.isfile(RFILE):
    raw = numpy.loadtxt(RFILE, delimiter=",", unpack=True, dtype=str)
    qdata = {}
    for i in range(raw.shape[0]):
        var = raw[i,0]
        val = raw[i,1:]
        try:
            val = val.astype(numpy.float64)
        except:
            pass
        qdata[var] = val

    for condition in ["disgust", "healthy", "unhealthy"]:
        for rating_type in ["pleasant", "desire", "disgust", "health", \
            "familiar"]:
            var = "{}_{}".format(condition, rating_type)
            dem[var] = numpy.zeros(dem["ppnr"].shape, dtype=qdata[var].dtype)
            for i, ppname in enumerate(dem["ppnr"]):
                ppi = list(qdata["ppname"]).index(ppname)
                dem[var][i] = qdata[var][ppi]

    for rating_type in ["pleasant", "desire", "disgust", "health", "familiar"]:
        dem["{}_avg".format(rating_type)] = numpy.nanmean(numpy.vstack( \
            [dem["{}_{}".format(condition, rating_type)] for condition \
            in ["disgust", "healthy", "unhealthy"]]), axis=0)

# Load rating data for individual stimuli.
if os.path.isfile(RSFILE):
    raw = numpy.loadtxt(RSFILE, delimiter=",", unpack=True, dtype=str)
    qdata = {}
    for i in range(raw.shape[0]):
        var = raw[i,0]
        val = raw[i,1:]
        try:
            val = val.astype(numpy.float64)
        except:
            pass
        qdata[var] = val

    for condition in ["disgust", "healthy", "unhealthy"]:
        for rating_type in ["pleasant", "desire", "disgust", "health", \
            "familiar"]:
            for stim_nr in range(1,16):
                var = "{}_{}_{}".format(condition, rating_type, stim_nr)
                dem[var] = numpy.zeros(dem["ppnr"].shape, \
                    dtype=qdata[var].dtype)
                for i, ppname in enumerate(dem["ppnr"]):
                    ppi = list(qdata["ppname"]).index(ppname)
                    dem[var][i] = qdata[var][ppi]

# Create a mask for exclusions and egg-only exclusions.
exclude = numpy.zeros(ppnames.shape[0], dtype=bool)
for i, ppname in enumerate(ppnames):
    if numpy.sum(numpy.isnan(signal_power_ch[i,:,:,:])) \
        == signal_power_ch[i,:,:,:].size:
        exclude[i] = True

with open(os.path.join(SUMDIR, "inclusions.csv"), "w") as f:
    f.write(",".join(["ppname", "gender", "age", "ethnicity", \
        "included_ton", "included_rating", "included_egg"]))
    for ppi, ppname in enumerate(ppnames):
        line = [ppname, dem["gender"][ppi], dem["age_in_years"][ppi], \
            dem["ethnicity"][ppi], \
            numpy.invert(numpy.isnan(dem["ton"][ppi])), \
            numpy.invert(numpy.isnan(dem["health_avg"][ppi])), \
            numpy.invert(numpy.isnan(numpy.nanmean(numpy.nanmean( \
                signal_power_ch[ppi,:,:,9], axis=0), axis=0))), \
            ]
        f.write("\n" + ",".join(map(str, line)))


# # # # #
# FIGURES

# Create a figure for the ratings. Figure should show the ratings separated
# by condition. Rating types are enumerated along the x-axis, and different
# the three conditions are shown side-by-side for each rating.
fig, ax = pyplot.subplots(figsize=(8.27,3.2), dpi=300)
fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.26)
x_pos = numpy.linspace(1, 6, 5)
x_jitter = numpy.array([-0.3, 0, 0.3])
v_width = 0.3
b_width = 0.2
rating_types = ["health", "disgust", "familiar", "pleasant", "desire"]
conditions = ["healthy", "unhealthy", "disgust"]
xticklabels = ["Healthiness", "Disgust", "Familiarity", "Pleasantness", \
    "Desire\nto eat"]
lines = {}
for ri, rating_type in enumerate(rating_types):
    for ci, condition in enumerate(conditions):
        var = "{}_{}".format(condition, rating_type)
        # Plot a kernel density estimate to indicate the distribution.
        v = ax.violinplot(100*dem[var], positions=[x_pos[ri]+x_jitter[ci]], \
            widths=[v_width], showmeans=False, showextrema=False, \
            showmedians=False)
        v["bodies"][0].set_color(COL[condition])
        # Plot a box plot on top of the distribution.
        b = ax.boxplot(100*dem[var], positions=[x_pos[ri]+x_jitter[ci]], \
            sym=".")
        b["medians"][0].set_linewidth(3)
        b["medians"][0].set_color(COL[condition])
        b["medians"][0].set_alpha(0.8)
        if condition not in lines.keys():
            lines[condition] = b["medians"][0]
            if condition == "disgust":
                lbl = "Disgusting"
            else:
                lbl = condition.capitalize()
            lines[condition].set_label(lbl)
# Finish, save, and close the plot.
ax.set_xlim([x_pos[0]+x_jitter[0]-v_width*0.6, \
    x_pos[-1]+x_jitter[-1]+v_width*0.6 + 1.2])
ax.set_xticks(x_pos)
ax.set_xticklabels(xticklabels, fontsize=11, rotation=20)
ax.set_xlabel("Self-reported food ratings", fontsize=14, fontweight="bold")
ax.set_ylim(-1, 101)
ax.set_ylabel("Rating (%)", fontsize=14, fontweight="bold")
handles = [lines[con] for con in conditions]
ax.legend(handles=handles, loc="upper right", fontsize=11, \
    title="Food type", title_fontproperties={"size":12, "weight":"bold"})
fig.savefig(os.path.join(SUMDIR, "ratings.png"))
pyplot.close(fig)

# Create a figure that shows EGG power as a function of TON score. The figure
# should have one window per condition.
conditions = ["disgust", "healthy", "unhealthy"]
legend_handle_order = ["healthy", "unhealthy", "disgust"]
sort_variables = ["ton", "health_avg"]
descriptors = ["lower", "middle", "higher"]
n_variables = len(sort_variables)
n_bins = len(descriptors)
fig, axes = pyplot.subplots(figsize=(2.76*n_bins, n_variables*3.5), dpi=300, \
    ncols=n_bins, nrows=n_variables, sharey=True, sharex=True)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.08, \
    hspace=0.25, wspace=0.07)
# Run through all cells of the figure.
for vi, sort_variable in enumerate(sort_variables):
    # Divide the scoring variable into bins.
    var_sorted_indices = numpy.argsort(dem[sort_variable])
    var_sorted = numpy.copy(dem[sort_variable])[var_sorted_indices]
    n = numpy.sum(numpy.invert(numpy.isnan(var_sorted)))
    n_per_bin = numpy.ones(n_bins, dtype=int) * (n//n_bins)
    n_per_bin[:n%n_bins] += 1
    print("N per bin: {} (sorted on {})".format(n_per_bin, sort_variable))
    bins = []
    si = 0
    for i in range(n_bins):
        ei = si + n_per_bin[i]
        bins.append(var_sorted_indices[si:ei])
        si = copy.copy(ei)
    # For within-participant error bars, compute corrected values. (Please 
    # excuse the reshapes; I temporarily forgot that "keepdims" exists.)
    m_channels = numpy.nanmean(signal_power_ch, axis=1)
    n_participants, n_conditions, n_samples = m_channels.shape
    cpm_normogastric = (freq_cpm > 2) & (freq_cpm < 4)
    m_channels /= numpy.nanmax( \
        m_channels[:,CONDITIONS.index("baseline"),cpm_normogastric], \
        axis=1).reshape(n_participants,1,1)
    m_subjects = numpy.nanmean(m_channels, axis=1).reshape( \
        n_participants, 1, n_samples)
    m_grand = numpy.nanmean(m_subjects, axis=0).reshape(1,1,n_samples)
    nv = m_channels - m_subjects + m_grand
    # Loop through all TON bins.
    for ai, bi in enumerate(bins):
        # Grab the axes to draw in.
        ax = axes[vi,ai]
        # Set the title for this axes.
        if sort_variable == "ton":
            tit = "{} orthorexia\n".format(descriptors[ai].capitalize()) \
                + r"($M_{TON}$" + "={:.1f}, {:.0f}-{:.0f})".format( \
                numpy.nanmean(dem[sort_variable][bi]), \
                numpy.nanmin(dem[sort_variable][bi]), \
                numpy.nanmax(dem[sort_variable][bi]), \
                )
        elif sort_variable == "health_avg":
            tit = "{} health ratings\n".format(descriptors[ai].capitalize()) \
                + r"($M$" + "={:.2f}, {:.2f}-{:.2f})".format( \
                numpy.nanmean(dem[sort_variable][bi]), \
                numpy.nanmin(dem[sort_variable][bi]), \
                numpy.nanmax(dem[sort_variable][bi]), \
                )
        else:
            tit = "{} {}\n".format(descriptors[ai].capitalize(), sort_variable) \
                + r"($M$" + "={:.2f}, {:.2f}-{:.2f})".format( \
                numpy.nanmean(dem[sort_variable][bi]), \
                numpy.nanmin(dem[sort_variable][bi]), \
                numpy.nanmax(dem[sort_variable][bi]), \
                )
        ax.set_title(tit, fontsize=13)
        # Loop through the conditions.
        lines = {}
        for condition in conditions:
            # Find the data index for this condition.
            ci = CONDITIONS.index(condition)
            # Average across participants in this bin.
            m = numpy.nanmean(m_channels[bi,ci,:], axis=0)
            sem = numpy.nanstd(nv[bi,ci,:], axis=0) / numpy.sqrt(n_per_bin[ai])
            # Plot the line and error margins.
            lines[condition] = ax.plot(freq_cpm, m, marker=None, ls="-", lw=3, \
                label=condition.capitalize(), color=COL[condition], alpha=0.8)
            ax.fill_between(freq_cpm, m-sem, m+sem, color=COL[condition], \
                alpha=0.2)
        # Add x ticks.
        ax.set_xticks(numpy.arange(1, 10, 2))
    # Add y label
    axes[vi,0].set_ylabel("EGG power (scaled)", fontsize=13, fontweight="bold")
    #Add legend, but only to the top plot.
    if vi == 0:
        handles = [lines[con][0] for con in legend_handle_order]
        axes[0,-1].legend(handles=handles, loc="upper right", fontsize=10, \
            title="Food type", title_fontproperties={"size":12, "weight":"bold"})
axes[n_variables-1, 1].set_xlabel("Frequency (cycles/minute)", fontsize=13, \
    fontweight="bold")
fig.savefig(os.path.join(SUMDIR, "EGG_fft.png"))
pyplot.close(fig)


# # # # #
# LME SELF-REPORT

# Set the LME up.
formula = "rating_{} ~ condition + ton + condition*ton"
re_formula = "1"
vc_formula = {"stim_nr": "0 + C(stim_nr)"}
included_conditions = ["healthy", "unhealthy", "disgust"]

# Create a dataframe to use with the LMEs.
n_participants = signal_power_ch.shape[0]
n_conditions = len(included_conditions)
n_stimuli = 15
shape = (n_participants, n_conditions, n_stimuli)
pps = numpy.zeros(shape, dtype=ppnames.dtype)
con = numpy.zeros(shape, dtype="<U9")
stim_nr = numpy.zeros(shape, dtype="<U2")
ton = numpy.zeros(shape, dtype=dem["ton"].dtype)
for pi in range(shape[0]):
    pps[pi,:,:] = ppnames[pi]
    ton[pi,:,:] = dem["ton"][pi]
for ci, condition in enumerate(included_conditions):
    con[:,ci,:] = condition
for si in range(n_stimuli):
    stim_nr[:,:,si] = str(si+1).rjust(2, "0")
r = {}
for rating_type in ["disgust", "health", "desire", "familiar", "pleasant"]:
    r[rating_type] = numpy.zeros(shape, dtype=numpy.float64)
    for ci, condition in enumerate(included_conditions):
        for si in range(n_stimuli):
            var = "{}_{}_{}".format(condition, rating_type, si+1)
            if var in dem.keys():
                r[rating_type][:,ci,si] = dem[var]
            else:
                r[rating_type][:,ci,si] = numpy.nan
df = pandas.DataFrame()
df["ppname"] = pps.flatten()
df["condition"] = con.flatten()
df["condition"][df["condition"]=="healthy"] = "_healthy"
df["stim_nr"] = stim_nr.flatten()
df["ton"] = ton.flatten()
for rating_type in r.keys():
    df["rating_{}".format(rating_type)] = r[rating_type].flatten()
for var in ["ton"] + ["rating_{}".format(rating_type) for rating_type in r.keys()]:
    df[var] = (df[var]-numpy.nanmean(df[var])) / numpy.nanstd(df[var])

# Run through all rating types.
for rating_type in ["disgust", "health", "desire", "familiar", "pleasant"]:

    # Run the LME.
    fpath = os.path.join(LMEDIR, \
        "preregistered_{}_rating_lme.txt".format(rating_type))
    with open(fpath, "w") as f:
        f.write("Self-reported {} ratings".format(rating_type))
    
    print("Running {}: {}".format(rating_type, formula))
    lme = MixedLM.from_formula(formula.format(rating_type), df, \
        groups="ppname", re_formula=re_formula, vc_formula=vc_formula, \
        missing="drop")
    result = lme.fit()
    
    # Write the result to file.
    with open(fpath, "a") as f:
        f.write("\n\n")
        f.write(result.summary().as_text())
        f.write(formula.format(rating_type))
        f.write("\nAIC = {}; BIC = {}".format( \
            aic(result.llf, result.nobs, result.df_modelwc), \
            bic(result.llf, result.nobs, result.df_modelwc)))    
        f.write("\n\n")


# # # # #
# LME EGG

# Set the LME up.
formula = "signal ~ condition + freq + ton + condition*freq*ton"
re_formula = "1"
vc_formula = {"channel": "0 + C(channel)"}
included_conditions = copy.deepcopy(CONDITIONS)

# Get the required values in a dataframe.
n_participants = signal_power_ch.shape[0]
n_channels = signal_power_ch.shape[1]
n_freqs = signal_power_ch.shape[-1]
n_conditions = len(included_conditions)
shape = (n_participants, n_channels, n_conditions, n_freqs)
pps = numpy.zeros(shape, dtype=ppnames.dtype)
channel = numpy.zeros(shape, dtype=int)
frequency = numpy.zeros(shape, dtype=float)
con = numpy.zeros(shape, dtype="<U9")
ton = numpy.zeros(shape, dtype=dem["ton"].dtype)
health = numpy.zeros(shape, dtype=dem["health_avg"].dtype)
for pi in range(shape[0]):
    pps[pi,:,:,:] = ppnames[pi]
    ton[pi,:,:,:] = dem["ton"][pi]
    for ci, condition in enumerate(included_conditions):
        if condition == "baseline":
            health[pi,:,ci,:] = dem["health_avg"][pi]
        else:
            health[pi,:,ci,:] = dem["{}_health".format(condition)][pi]
for ch in range(n_channels):
    channel[:,ch,:,:] = ch
for ci, condition in enumerate(included_conditions):
    con[:,:,ci,:] = condition
for fi in range(n_freqs):
    frequency[:,:,:,fi] = freq_cpm[fi]

s = numpy.zeros(shape, dtype=signal_power_ch.dtype)
for ci in range(n_conditions):
    ci_ = CONDITIONS.index(included_conditions[ci])
    for fi in range(n_freqs):
        s[:,:,ci,fi] = signal_power_ch[:,:,ci_,fi]

df = pandas.DataFrame()
df["ppname"] = pps.flatten()
df["channel"] = channel.flatten()
df["condition"] = con.flatten()
df["freq"] = frequency.flatten()
df["ton"] = ton.flatten()
df["health_rating"] = health.flatten()
df["signal"] = s.flatten()

# Standardise continuous data.
for var in ["freq", "ton", "health_rating", "signal"]:
    df[var] = (df[var]-numpy.nanmean(df[var])) / numpy.std(df[var])

# Run the LME.
with open(os.path.join(LMEDIR, "preregistered_egg_lme.txt"), "w") as f:
    f.write("Pre-registered analysis")

print("Running {}".format(formula))

# Run a linear-mixed effects model.
lme = MixedLM.from_formula(formula, df, groups="ppname", \
    re_formula=re_formula, vc_formula=vc_formula, missing="drop")
result = lme.fit()

# Write the result to file.
with open(os.path.join(LMEDIR, "preregistered_egg_lme.txt"), "a") as f:
    f.write("\n\n")
    f.write(result.summary().as_text())
    f.write(formula.format(rating_type))
    f.write("\nAIC = {}; BIC = {}".format( \
        aic(result.llf, result.nobs, result.df_modelwc), \
        bic(result.llf, result.nobs, result.df_modelwc)))    
    f.write("\n\n")


# Set the LMEs up.
formulae = [ \
    # Exploratory analysis: same as pre-registered, but without baseline.
    "signal ~ condition + freq + ton + condition*freq*ton", \
    # Exploratory analysis: using health ratings instead of TON-17.
    "signal ~ condition + freq + health_rating + condition*freq*health_rating", \
    ]
re_formula = "1"
vc_formula = {"channel": "0 + C(channel)"}
included_conditions = ["healthy", "unhealthy", "disgust"]

# Get the required values in a dataframe.
n_participants = signal_power_ch.shape[0]
n_channels = signal_power_ch.shape[1]
n_freqs = signal_power_ch.shape[-1]
n_conditions = len(included_conditions)
shape = (n_participants, n_channels, n_conditions, n_freqs)
pps = numpy.zeros(shape, dtype=ppnames.dtype)
channel = numpy.zeros(shape, dtype=int)
frequency = numpy.zeros(shape, dtype=float)
con = numpy.zeros(shape, dtype="<U9")
ton = numpy.zeros(shape, dtype=dem["ton"].dtype)
health = numpy.zeros(shape, dtype=dem["health_avg"].dtype)
for pi in range(shape[0]):
    pps[pi,:,:,:] = ppnames[pi]
    ton[pi,:,:,:] = dem["ton"][pi]
    for ci, condition in enumerate(included_conditions):
        health[pi,:,ci,:] = dem["{}_health".format(condition)][pi]
for ch in range(n_channels):
    channel[:,ch,:,:] = ch
for ci, condition in enumerate(included_conditions):
    con[:,:,ci,:] = condition
for fi in range(n_freqs):
    frequency[:,:,:,fi] = freq_cpm[fi]

s = numpy.zeros(shape, dtype=signal_power_ch.dtype)
for ci in range(n_conditions):
    ci_ = CONDITIONS.index(included_conditions[ci])
    for fi in range(n_freqs):
        s[:,:,ci,fi] = signal_power_ch[:,:,ci_,fi]

df = pandas.DataFrame()
df["ppname"] = pps.flatten()
df["channel"] = channel.flatten()
df["condition"] = con.flatten()
df["condition"][df["condition"]=="healthy"] = "_healthy"
df["freq"] = frequency.flatten()
df["ton"] = ton.flatten()
df["health_rating"] = health.flatten()
df["signal"] = s.flatten()

# Standardise continuous data.
for var in ["freq", "ton", "health_rating", "signal"]:
    df[var] = (df[var]-numpy.nanmean(df[var])) / numpy.std(df[var])

# Run all LMEs.
with open(os.path.join(LMEDIR, "exploratory_egg_lme.txt"), "w") as f:
    f.write("Exploratory analysis")

for formula in formulae:
    
    print("Running {}".format(formula))
    
    # Run a linear-mixed effects model.
    lme = MixedLM.from_formula(formula, df, groups="ppname", \
        re_formula=re_formula, vc_formula=vc_formula, missing="drop")
    result = lme.fit()
    
    # Write the result to file.
    with open(os.path.join(LMEDIR, "exploratory_egg_lme.txt"), "a") as f:
        f.write("\n\n")
        f.write(result.summary().as_text())
        f.write(formula.format(rating_type))
        f.write("\nAIC = {}; BIC = {}".format( \
            aic(result.llf, result.nobs, result.df_modelwc), \
            bic(result.llf, result.nobs, result.df_modelwc)))    
        f.write("\n\n")
