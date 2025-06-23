#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import cv2
import matplotlib
from matplotlib import pyplot
import numpy
from scipy.stats import ttest_ind


def buf_count_newlines_gen(fpath):
    # Fastest counter from: https://stackoverflow.com/questions/845058/
    # how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b
    with open(fpath, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count

def read_facial_landmarks(fpath):
    # Read number of newlines in file, which should be exactly the number of
    # data lines (the final line does not end on a newline).
    n_lines = buf_count_newlines_gen(fpath)
    
    if n_lines < 2:
        return None, None, None, None, None

    # Load data from file.
    with open(fpath, "r") as f:
        
        # Get the header out first.
        header = f.readline()
        
        # Parse the header to count the number of emotions and landmarks.
        header = header.rstrip("\n").split(",")
        
        # Find the emotions, and get the number of landmarks.
        n_points = 0
        emotion_names = []
        for var in header:
            if var in ["count", "time"]:
                continue
            elif var[:2] in ["x_", "y_", "z_"]:
                axis, nr = var.split("_")
                if nr.isnumeric():
                    if int(nr) > n_points:
                        n_points = int(nr) + 1
            else:
                emotion_names.append(var)
        
        # Find indices for variables we want to store.
        i_count = header.index("count")
        i_timestamp = header.index("time")
        if n_points > 0:
            i_x = header.index("x_0")
            i_y = header.index("y_0")
            i_z = header.index("z_0")
        if len(emotion_names) > 0:
            i_emotion = {}
            for name in emotion_names:
                i_emotion[name] = header.index(name)
        
        # Create empty numpy arrays.
        count = numpy.zeros(n_lines, dtype=numpy.int64)
        timestamp = numpy.zeros(n_lines, dtype=numpy.float64)
        landmarks = numpy.zeros((3, n_points, n_lines), dtype=numpy.float64)
        emotions = {}
        for name in emotion_names:
            emotions[name] = numpy.zeros(n_lines, dtype=numpy.float64)
        messages = []

        # Now read all data.
        msg_lines = []
        for i, line in enumerate(f):
            # Strip newline, and split by commas.
            line = line.rstrip("\n").split(",")
            # Check if this is a MSG line.
            if line[0] == "MSG":
                t = float(line[1])
                msg = line[2]
                msg_lines.append(i)
                messages.append((t, msg))
            # All other lines are data.
            else:
                count[i] = int(line[i_count])
                timestamp[i] = float(line[i_timestamp])
                if n_points > 0:
                    landmarks[0,:,i] = line[i_x:i_x+n_points]
                    landmarks[1,:,i] = line[i_y:i_y+n_points]
                    landmarks[2,:,i] = line[i_z:i_z+n_points]
                if len(emotion_names) > 0:
                    for name in i_emotion.keys():
                        emotions[name][i] = line[i_emotion[name]]

    # Filter out the message lines from the data arrays.
    include = numpy.ones(n_lines, dtype=bool)
    include[msg_lines] = False

    count = count[include]
    timestamp = timestamp[include]
    landmarks = landmarks[:,:,include]
    for name in emotions.keys():
        emotions[name] = emotions[name][include]

    return count, timestamp, landmarks, emotions, messages

def read_trial_info(fpath, int_vars=[], float_vars=[]):
    info = {}
    raw = numpy.loadtxt(fpath, dtype=str, delimiter="\t")
    for i in range(raw.shape[1]):
        var = raw[0,i]
        val = raw[1:,i]
        if var in int_vars:
            val = val.astype(numpy.int64)
        elif var in float_vars:
            val = val.astype(numpy.float64)
        info[var] = val
    
    return info

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


# ANALYSIS SETTINGS

# Set the baseline method.
BASELINE_METHOD = "difference"
if BASELINE_METHOD is not None:
    BASELINE_SAMPLES = 3
else:
    BASELINE_SAMPLES = 0

# Set the number of samples extracted from the face recording after stimulus 
# onset. Set to None to extract all samples.
STIMULUS_SAMPLES = 13

# Choose which points to compute transformations on. (Doing the whole face 
# leads to misalignments, but using a subset from strategically chosen places 
# works much better.)
TRANSFORMATION_POINTS = numpy.array([4, 337, 108, 376, 147, 140, 369])

# EXPERIMENT SETTINGS
# Conditions used in the experiment for which we'd like to extract facial 
# landmarks.
CONDITIONS = ["disgust", "healthy", "unhealthy"]
# Number of stimuli and how often they repeat.
N_STIMULI_PER_CONDITION = 15
N_STIMULUS_REPEATS = 2
# Equipment settings.
SAMPLING_RATE = 10.0

# Variable names in the trial info file that should be cast as int or float.
INFO_INT_VARS = ["trialnr", "img_nr", "stim_trigger"]
INFO_FLOAT_VARS = ["fix_onset", "stim_onset", "stim_offset"]

# Plot colours.
COL = {}
COL["disgust"] = "#8f5902"
COL["healthy"] = "#4e9a06"
COL["unhealthy"] = "#c4a000"
COL["baseline"] = "#204a87"

# FILES AND FOLDERS
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data")
OUTDIR = os.path.join(DIR, "output_faces")
TMPDIR = os.path.join(OUTDIR, "all_reduced_data")
for dir_path in [OUTDIR, TMPDIR]:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
CANONICAL_FACE = os.path.join(DIR, "canonical_face_model.csv")


# # # # #
# RUN ANALYSIS

# Load the canonical face data.
canonical_face_landmarks = numpy.loadtxt(CANONICAL_FACE, \
    delimiter=",", unpack=True)
canonical_face_landmarks[1,:] *= -1
set_point_origin(canonical_face_landmarks)
scale_points(canonical_face_landmarks)

# Find all face data.
all_fnames = []
for fname in os.listdir(DATADIR):
    # Check if the data type is facial landmarks.
    if "_face.csv" in fname:
        all_fnames.append(fname)
all_fnames.sort()
print("Located {} facial landmark recordings".format(len(all_fnames)))

# Create memory maps to hold extracted data.
unique_ppnames = numpy.unique(numpy.array( \
    [fname.split("_")[0] for fname in all_fnames], dtype="<U5"))
ppnames = numpy.memmap(os.path.join(TMPDIR, "face_ppnames.dat"), mode="w+", \
    shape=unique_ppnames.shape, dtype=unique_ppnames.dtype)
ppnames[:] = unique_ppnames

n_samples = N_STIMULI_PER_CONDITION * N_STIMULUS_REPEATS
if STIMULUS_SAMPLES is None:
    n_samples *= 100
else:
    n_samples *= STIMULUS_SAMPLES
n_samples = round(n_samples * 1.1)
shape = (ppnames.shape[0], len(CONDITIONS), 3, 468, n_samples)
facial_landmarks = numpy.memmap(os.path.join(TMPDIR, "face_landmarks.dat"), \
    mode="w+", shape=shape, dtype=numpy.float32)
facial_landmarks[:] = numpy.nan

shape = (ppnames.shape[0], len(CONDITIONS), n_samples)
stimulus_image = numpy.memmap(os.path.join(TMPDIR, "stimulus_image.dat"), \
    mode="w+", shape=shape, dtype="<U20")


# Loop through all detected facial-landmark files.
print("Processing data from {} files.".format(len(all_fnames)))
for fi, fname in enumerate(all_fnames):
    
    # Parse the file name to construct the name of the associated info file.
    name, ext = os.path.splitext(fname)
    ppname, _ = name.split("_")
    ppi = list(ppnames).index(ppname)
    fname_info = name.replace("_face", "") + ".txt"
    
    # Construct the path to the current file.
    fpath = os.path.join(DATADIR, fname)
    # Construct the path to the info file.
    fpath_info = os.path.join(DATADIR, fname_info)
    
    # Load the data.
    print("\tLoading file {}...".format(fname))
    t0 = time.time()
    c, t, lm, emo, msg = read_facial_landmarks(fpath)
    t1 = time.time()
    print("\tLoaded in {:.2f} seconds".format(t1-t0))
    
    if lm is None:
        print("\tFile is empty; skipping!")
        continue
    
    # Load the trial info.
    trial_info = read_trial_info(fpath_info, int_vars=INFO_INT_VARS, \
        float_vars=INFO_FLOAT_VARS)
    
    # Remove the eyes.
    lm = lm[:,:468,:]

    # Compute average landmark positions, then attempt to align these with 
    # the canonical face.
    set_point_origin(lm)
    scale_points(lm)
    print("\tAligning landmarks to canonical face...")
    t0 = time.time()
    target = canonical_face_landmarks[:,TRANSFORMATION_POINTS].T
    for sample in range(lm.shape[2]):
        success, transformation, scale = cv2.estimateAffine3D( \
            lm[:,TRANSFORMATION_POINTS,sample].T, target, True)
        if success:
            lm[:,:,sample] = (numpy.dot(lm[:,:,sample].T, \
                transformation[:,:3].T) + transformation[:,3]).T
        else:
            print("\t\tFailed to align sample {}".format(sample))
            lm[:,:,sample] = numpy.nan
    t1 = time.time()
    print("\tAlignment finished in {:.2} seconds!".format(t1-t0))

    # Find the onsets in the data.
    onsets = []
    offsets = []
    for m in msg:
        if m[1] == "STIMULUS_ONSET":
            onsets.append(m[0])
        elif m[1] == "STIMULUS_OFFSET":
            offsets.append(m[0])

    # Convert onsets/offsets into NumPy arrays.
    onsets = numpy.array(onsets)
    offsets = numpy.array(offsets)
    
    # Check if we have the anticipated numbers of onsets and offsets.
    if (onsets.shape[0] == offsets.shape[0]) \
        and (onsets.shape[0] == trial_info["trialnr"].shape[0]):
        print("\tAll expected onsets and offsets found.")
    else:
        print("\tSkipping participant due to missing onsets or offsets.")
        continue
    
    # Ensure there are both disgust and neutral trials.
    trial_types = numpy.unique(trial_info["img_category"])
    trial_type_missing = False
    for trial_type in CONDITIONS:
        if trial_type not in trial_types:
            print("\tMissing stimulus category '{}'".format(trial_type))
            trial_type_missing = True
    if trial_type_missing:
        print("\tSkipping participant due to missing trial types.")
        continue
    
    # Create selections for the disgust trials, and for the neutral trials.
    sel = {}
    sel_samples = {}
    stim_samples = {}
    d_lm = numpy.zeros(lm.shape, lm.dtype) * numpy.nan
    for trial_type in trial_types:
        sel[trial_type] = trial_info["img_category"] == trial_type
        sel_samples[trial_type] = numpy.zeros(t.shape, dtype=bool)
        stim_samples[trial_type] = numpy.zeros(t.shape, \
            dtype=stimulus_image.dtype)
        for i, onset_time in enumerate(onsets):
            if sel[trial_type][i]:
                si = numpy.argmin(numpy.abs(t - onset_time))
                bi = si - BASELINE_SAMPLES
                if STIMULUS_SAMPLES is None:
                    ei = numpy.argmin(numpy.abs(t - offsets[i]))
                else:
                    ei = si + STIMULUS_SAMPLES
                sel_samples[trial_type][si:ei] = True
                # Get the image name.
                img_fname = trial_info["stim_img"][i]
                img_name, ext = os.path.splitext(img_fname)
                stim_samples[trial_type][si:ei] = img_name
                # Compute mean landmarks in the baseline.
                baseline = numpy.nanmean(lm[:,:,bi:si], axis=2)
                # Compute distance from baseline.
                d_lm[:,:,si:ei] = lm[:,:,si:ei] \
                    - baseline.reshape(lm.shape[0],lm.shape[1],1)
    
    # Separate samples by condition.
    diff = {}
    landmarks = {}
    stim_names = {}
    for trial_type in trial_types:
        diff[trial_type] = d_lm[:,:,sel_samples[trial_type]]
        landmarks[trial_type] = lm[:,:,sel_samples[trial_type]]
        stim_names[trial_type] = \
            stim_samples[trial_type][sel_samples[trial_type]]
    
    # Save in memory-mapped array.
    print("\tStoring data in memory-mapped array.")
    for ti, trial_type in enumerate(CONDITIONS):
        ei = diff[trial_type].shape[-1]
        facial_landmarks[ppi, ti, :, :, :ei] = diff[trial_type]
        stimulus_image[ppi, ti, :ei] = stim_names[trial_type]
