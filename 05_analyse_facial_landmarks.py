#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import matplotlib
from matplotlib import pyplot
import numpy
import pandas
import scipy.stats
from statsmodels.api import MixedLM


def set_point_origin(landmarks, origin_point=4):
    # Set landmark 4 (tip of the nose) as the origin (0,0).
    if len(landmarks.shape) == 3:
        landmarks -= landmarks[:,origin_point:origin_point+1,:]
    else:
        landmarks -= landmarks[:,origin_point:origin_point+1]

def scale_points(landmarks, scale="face"):
    # Normalise width on basis of eye-to-eye distance.
    if scale == "eyes":
        if len(landmarks.shape) == 3:
            f = 1.0 / numpy.abs(landmarks[0,133,:] - landmarks[0,362,:])
        else:
            f = 1.0 / numpy.abs(landmarks[0,133] - landmarks[0,362])
    # Normalise width on the basis of distance between left and right sides 
    # of the face (points 162 and 389).
    elif scale == "face":
        if len(landmarks.shape) == 3:
            f = 1.0 / numpy.abs(landmarks[0,162,:] - landmarks[0,389,:])
        else:
            f = 1.0 / numpy.abs(landmarks[0,162] - landmarks[0,389])
    else:
        raise Exception(("Unknown scaling option '{}'; choose 'eyes' or "  \
            + "'face").format(scale))
    landmarks *= f

def plot_facial_landmarks_3d(fpath, arr_s, arr_d, ax=None, elevation=None, \
    azimuth=None, roll=None, marker_size=None, cmap="viridis", vlim=None, \
    panel_size=(6.0,6.0), dpi=100.0):

    # Compute Euclidean distance between arrow start and ends.
    arr_dist = numpy.sqrt(numpy.sum(arr_d**2, axis=0))
    
    # Get the colour map and set limits on it.
    cmap = matplotlib.cm.get_cmap(cmap)
    if vlim is None:
        _vlim = max(numpy.abs(numpy.nanmin(arr_dist)), \
            numpy.abs(numpy.nanmax(arr_dist)))
        vlim = [-_vlim, _vlim]
    norm = matplotlib.colors.Normalize(vmin=vlim[0], vmax=vlim[1])
    colour = cmap(norm(arr_dist))[:,:3]
    
    # These are magic numbers that work with the canonical face model.
    if elevation is None:
        elevation = [ \
            [135, 120, 135], \
            [0, 90, 0], \
            [45, 60, 45]]
    if azimuth is None:
        azimuth = [ \
            [135, 90, 45], \
            [0, 90, 180], \
            [45, 90, 135]]
    if roll is None:
        roll = [ \
            [45, 0, 315], \
            [270, 0, 90], \
            [315, 0, 45]]
    
    if marker_size is None:
        marker_size = 5

    # Create a new figure. Numbers or rows and columns come from elevation
    # only. Ideally, we'd do a check to see if the dimensions match.
    n_cols = len(elevation)
    n_rows = len(elevation[0])
    if ax is None:
        fig, axes = pyplot.subplots( \
            figsize=(n_cols*panel_size[0],n_rows*panel_size[1]), dpi=dpi, \
            nrows=n_rows, ncols=n_cols, subplot_kw={"projection":"3d"})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, \
            hspace=0, wspace=0)
    else:
        fig = None
        axes = None
    # Loop through all panels.
    for row in range(n_rows):
        for col in range(n_cols):
            # Select the current axis to draw in.
            if axes is not None:
                ax = axes[col,row]
            # Set the view to the specific elevation, azimuth, and roll.
            ax.view_init(elev=elevation[col][row], azim=azimuth[col][row], \
                roll=roll[col][row])
            # Remove the ugly grey background.
            ax.set_axis_off()
            # Scatterplot for the arrows' starting positions.
            ax.scatter(arr_s[0,:], arr_s[1,:], arr_s[2,:], \
                marker="o", s=marker_size, c="black", alpha=0.2)
            
            # NOTE: The following should work, but doesn't pass the colours
            # to the Line3DCollection properly. Hence, we'll need to draw the
            # stupid lines one-by-one, which takes for ever.
            # Draw arrows from the starting positions to the ending positions.
            # arrows = ax.quiver(arr_s[0,:], arr_s[1,:], arr_s[2,:], \
            #     arr_d[0,:], arr_d[1,:], arr_d[2,:], \
            #     color=colour, length=1, normalize=False)
            for i in range(arr_s.shape[1]):
                ax.quiver(arr_s[0,i], arr_s[1,i], arr_s[2,i], \
                    arr_d[0,i], arr_d[1,i], arr_d[2,i], \
                    color=colour[i,:], length=1, normalize=False)

            # # The following doesn't do a lot, but would be the settings for
            # # the canonical face.
            # ax.set_xlim(-0.52, 0.52)
            # ax.set_ylim(-0.60, 0.60)
            # ax.set_zlim(-0.67, 0.00)
    # Squeeze whitespace out of the figure.
    if fig is not None:
        fig.tight_layout(pad=-8, h_pad=-10, w_pad=-10)
    # Save and close, or return if no location was provided.
    if fpath is None:
        return fig, axes
    elif fig is None:
        return None, None
    else:
        fig.savefig(fpath)
        pyplot.close(fig)
        return None, None


# # # # #
# SETTINGS

# Set the baseline method.
BASELINE_METHOD = "difference"
if BASELINE_METHOD is not None:
    BASELINE_SAMPLES = 3
else:
    BASELINE_SAMPLES = 0

# Set the number of samples extracted from the face recording after stimulus 
# onset. Set to None to extract all samples.
STIMULUS_SAMPLES = 13

# Split data into train and test sub-samples for regression?
TRAIN_TEST_SPLIT = False

# Plot options.
PLOT_ONLY_SIG = True
MULTIPLE_COMPARISONS_CORRECTION = "bonferroni"
CREATE_INDIVIDUAL_PLOTS = False
CREATE_TIMELINE_VIDEO = False

# EXPERIMENT SETTINGS
# Conditions used in the experiment for which we'd like to extract facial 
# landmarks.
CONDITIONS = ["disgust", "healthy", "unhealthy"]
# Number of stimuli and how often they repeat.
N_STIMULI_PER_CONDITION = 15
N_STIMULUS_REPEATS = 2
# Equipment settings.
SAMPLING_RATE = 10.0

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

# FILES AND FOLDERS
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data")
DEMFILE = os.path.join(DATADIR, "demographics.csv")
QFILE = os.path.join(DATADIR, "data_questionnaires.csv")
OUTDIR = os.path.join(DIR, "output_faces_DEBUG")
TMPDIR = os.path.join(OUTDIR, "all_reduced_data")
IOUTDIR = os.path.join(OUTDIR, "individual_graphs")
VOUTDIR = os.path.join(IOUTDIR, "videos")
for dir_path in [OUTDIR, TMPDIR, IOUTDIR, VOUTDIR]:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
CANONICAL_FACE = os.path.join(DIR, "canonical_face_model.csv")


# # # # #
# LOAD DATA

# Load the canonical face data.
canonical_face_landmarks = numpy.loadtxt(CANONICAL_FACE, \
    delimiter=",", unpack=True)
canonical_face_landmarks[1,:] *= -1
set_point_origin(canonical_face_landmarks)
scale_points(canonical_face_landmarks)

# Plot the canonical phase with visible axes, so we can interpret directions.
cmap = matplotlib.cm.get_cmap("coolwarm")
norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
for i in range(canonical_face_landmarks.shape[0]):
    col = cmap(norm(canonical_face_landmarks[i,:]))
    fig, ax = pyplot.subplots(figsize=(6.0,6.0), dpi=100.0, nrows=1, ncols=1, \
        subplot_kw={"projection":"3d"})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    ax.view_init(elev=135, azim=135, roll=45)
    ax.scatter(canonical_face_landmarks[0,:], \
        canonical_face_landmarks[1,:], canonical_face_landmarks[2,:], \
        marker="o", s=5, c=col, alpha=0.2)
    pyplot.savefig(os.path.join(OUTDIR, "canonical_face_ax-{}.png".format(i)))
    pyplot.close(fig)

# Load participant names.
ppnames = numpy.memmap(os.path.join(TMPDIR, "face_ppnames.dat"), mode="r", \
    dtype="<U5")
# Load facial landmarks.
n_samples = N_STIMULI_PER_CONDITION * N_STIMULUS_REPEATS
if STIMULUS_SAMPLES is None:
    n_samples *= 100
else:
    n_samples *= STIMULUS_SAMPLES
n_samples = round(n_samples * 1.1)
shape = (ppnames.shape[0], len(CONDITIONS), 3, 468, n_samples)
facial_landmarks = numpy.copy(numpy.memmap(os.path.join(TMPDIR, "face_landmarks.dat"), \
    mode="r", shape=shape, dtype=numpy.float32))
# Load stimulus names.
shape = (ppnames.shape[0], len(CONDITIONS), n_samples)
stimulus_image = numpy.memmap(os.path.join(TMPDIR, "stimulus_image.dat"), \
    mode="r", shape=shape, dtype="<U20")

# Use stimulus image to determine stimulus repetitions.
stimulus_repetition = numpy.zeros(stimulus_image.shape, dtype=int)
for ppi in range(stimulus_image.shape[0]):
    for ci in range(stimulus_image.shape[1]):
        stimulus_names = numpy.unique(stimulus_image[ppi,ci,:])
        for name in stimulus_names:
            sel = stimulus_image[ppi,ci,:] == name
            if name == "":
                stimulus_repetition[ppi,ci,sel] = -1
            else:
                indices = numpy.where(sel)[0]
                stimulus_repetition[ppi,ci,indices[:STIMULUS_SAMPLES]] = 0
                stimulus_repetition[ppi,ci,indices[STIMULUS_SAMPLES:]] = 1

# Load demographic data.
if os.path.isfile(DEMFILE):
    raw = numpy.loadtxt(DEMFILE, delimiter=",", unpack=True, dtype=str, usecols=range(5))
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


# # # # #
# RUN ANALYSIS

# Construct arrays that we can use for linear mixed effects analyses.
shape = (ppnames.shape[0], len(CONDITIONS), facial_landmarks.shape[-1])
# shape = (ppnames.shape[0], len(CONDITIONS), 1)
pps = numpy.zeros(shape, dtype=ppnames.dtype)
con = numpy.zeros(shape, dtype="<U{}".format(max([len(con) for con in CONDITIONS])))
ton = numpy.zeros(shape, dtype=dem["ton"].dtype)
for i in range(pps.shape[0]):
    pps[i,:,:] = ppnames[i]
    ton[i,:,:] = dem["ton"][i]
for i in range(con.shape[1]):
    con[:,i,:] = CONDITIONS[i]
df = pandas.DataFrame()
df["ppname"] = pps.flatten()
df["stim"] = stimulus_image.flatten()
df["repeat"] = stimulus_repetition.flatten()
df["condition"] = con.flatten()
df["condition"][df["condition"]=="healthy"] = "_healthy"
df["ton"] = ton.flatten()
df["ton"] = (df["ton"]-numpy.nanmean(df["ton"])) / numpy.std(df["ton"])

# Describe the formula to run.
formula = "face ~ condition + ton + condition*ton"
re_formula = "1"
vc_formula = {"stim":"0 + C(stim)"}
fixed_effects = [ \
    "condition[T.disgust]", "condition[T.unhealthy]", \
        "ton", "condition[T.disgust]:ton", "condition[T.unhealthy]:ton", \
    ]
fpath = os.path.join(OUTDIR, "lme_condition_ton.csv")

# Write headers to output files.
if os.path.isfile(fpath):
    print("File for this formula already exists: {}".format(formula))
else:
    header = ["plane", "landmark"]
    for effect_name in fixed_effects:
        header.extend(["beta_{}".format(effect_name), \
            "bse_{}".format(effect_name)])
    with open(fpath, "w") as f:
        f.write(",".join(header))

    # Run through each plane.
    for pi, plane in enumerate(["horizontal", "vertical", "depth"]):
        # Run through each landmark.
        for i in range(468):
            # Get the landmarks for this point.
            lm = numpy.copy(facial_landmarks[:,:,pi,i,:])
            # lm = numpy.nanmean(facial_landmarks[:,:,pi,i,:], axis=2)
            # Construct a data frame for this landmark.
            df["face"] = lm.flatten()
            df["face"] = (df["face"]-numpy.nanmean(df["face"])) / numpy.std(df["face"])
    
            # Run through all formulae.
            print("Fitting LME for plane {} and landmark {}: {}".format( \
                plane, i, formula))
            t0 = time.time()
    
            # Run linear mixed effects analysis.
            try:
                lme = MixedLM.from_formula(formula, df, groups="ppname", \
                    re_formula=re_formula, vc_formula=vc_formula, missing="drop")
                result = lme.fit()
            except ValueError:
                print("\tNo variability in landmark!")
                result = None
            
            # Write the result to file.
            line = [plane, i]
            for effect_name in fixed_effects:
                if result is not None:
                    line.extend([result.params[effect_name], \
                        result.bse[effect_name]])
                else:
                    line.extend(["nan", "nan"])
            with open(fpath, "a") as f:
                f.write("\n" + ",".join(map(str, line)))
    
            t1 = time.time()
            print("\tThat took {:.3f} seconds.".format(t1-t0))


# # # # #
# PLOT

# Load the data from the CSV.
with open(fpath, "r") as f:
    header = f.readline()
header = header.replace("\n", "").split(",")
content = numpy.loadtxt(fpath, delimiter=",", dtype=str, skiprows=1)
data = {}
for i in range(content.shape[1]):
    if header[i] == "plane":
        i_plane = i
        dtype = str
    elif header[i] == "landmark":
        i_landmark = i
        dtype = int
    else:
        dtype = float
    data[header[i]] = content[:,i].astype(dtype)

# Create a figure to plot all faces in.
n_rows = 1
n_cols = len(fixed_effects)
panel_size = (6.0, 7.0)
fig_model, axes_model = pyplot.subplots( \
    figsize=(n_cols*panel_size[0],n_rows*panel_size[1]), dpi=300.0, \
    nrows=n_rows, ncols=n_cols, subplot_kw={"projection":"3d"})
fig_model.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, \
    hspace=0, wspace=0)
fig_model.tight_layout(pad=-8, h_pad=0, w_pad=-15)
elevation = [[70]]
azimuth = [[160]]
roll = [[69]]
# Set more human-readable names for fixed effects.
axes_titles = { \
    "condition[T.disgust]": "Disgust", \
    "condition[T.unhealthy]": "Unhealthy", \
    "ton": "Orthorexia", \
    "condition[T.disgust]:ton": "Disgust * orthorexia", \
    "condition[T.unhealthy]:ton": "Unhealthy * orthorexia", \
    }

# Plot all the effects.
for ai, effect_name in enumerate(fixed_effects):

    # Find betas for all planes and landmarks.
    beta = numpy.zeros((3,468), dtype=float)
    bse = numpy.zeros((3,468), dtype=float)
    beta_var = "beta_{}".format(effect_name)
    bse_var = "bse_{}".format(effect_name)
    for pi, plane in enumerate(["horizontal", "vertical", "depth"]):
        for i in range(468):
            sel = (data["plane"] == plane) & (data["landmark"] == i)
            if numpy.sum(sel) > 0:
                row = numpy.where(sel)[0][0]
                beta[pi,i] = data[beta_var][row]
                bse[pi,i] = data[bse_var][row]

    # Compute Z and p values for the loaded betas.
    beta_pooled = numpy.sqrt(beta[0,:]**2 + beta[1,:]**2 + beta[2,:]**2)
    bse_pooled = numpy.sqrt(bse[0,:]**2 + bse[1,:]**2 + bse[2,:]**2)
    z_pooled = beta_pooled / bse_pooled
    p_pooled = 2 * (1.0 - scipy.stats.norm.cdf(numpy.abs(z_pooled)))
    if MULTIPLE_COMPARISONS_CORRECTION is None:
        sig_pooled = p_pooled < 0.05
    elif MULTIPLE_COMPARISONS_CORRECTION == "bonferroni":
        sig_pooled = p_pooled < (0.05 / p_pooled.size)
    
    print("effect {}: {}/{} significant".format(effect_name, \
        numpy.sum(sig_pooled), p_pooled.size))
    
    # Do a 3D plot for the betas.
    scale_factor = 0.5
    vlim = (0, 0.075)
    arr_s = numpy.copy(canonical_face_landmarks)
    arr_d = numpy.copy(beta) * scale_factor
    marker_size = numpy.ones(beta_pooled.shape, dtype=float)
    marker_size[numpy.invert(sig_pooled)] *= 2
    marker_size[sig_pooled] *= 20
    # Plot in the full-model figure.
    plot_facial_landmarks_3d(None, arr_s, arr_d, \
        ax=axes_model[ai], elevation=elevation, azimuth=azimuth, roll=roll,
        marker_size=marker_size, cmap="magma", vlim=vlim)
    text_pos = ( \
        numpy.nanmedian(arr_s, axis=1)[0], \
        numpy.nanmin(arr_s, axis=1)[1], \
        numpy.nanmax(arr_s, axis=1)[2], \
        )
    axes_model[ai].text(text_pos[0], text_pos[1], text_pos[2], \
        axes_titles[effect_name], horizontalalignment="center", fontsize=18)

    # Plot full view (across 9 orientations) for this effect.
    fpath_3d = os.path.join(OUTDIR, "3D_lme-{}.png".format(effect_name))
    plot_facial_landmarks_3d(fpath_3d, arr_s, arr_d, marker_size=marker_size, \
        cmap="magma", vlim=vlim, panel_size=(6.0,6.0), dpi=300.0)

# Finish the full-model plot.
bax = fig_model.add_axes([0.4,0.1,0.2,0.015])
norm = matplotlib.colors.Normalize(vmin=vlim[0]/scale_factor, vmax=vlim[1]/scale_factor)
cbar = matplotlib.colorbar.ColorbarBase(bax, cmap="magma", norm=norm, \
    ticks=[0.0, 0.05, 0.1, 0.15], orientation='horizontal')
cbar.set_ticklabels([0.0, 0.05, 0.1, 0.15], fontsize=14)
cbar.set_label(r"$\beta$ coefficient vector", fontsize=18)
fig_model.savefig(os.path.join(OUTDIR, "3D_lme-full-model.png"))
pyplot.close(fig_model)
