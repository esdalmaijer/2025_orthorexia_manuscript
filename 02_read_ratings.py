#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy


# # # # #
# CONSTANTS

# Excluded participants
EXCLUDED = [ \
    ]

# Included conditions.
CONDITIONS = ["disgust", "healthy", "unhealthy"]
RATING_TYPES = ["pleasant", "desire", "disgust", "health", "familiar"]

# FILES AND FOLDERS
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data")
OUTDIR = os.path.join(DIR, "output_ratings")
TMPDIR = os.path.join(OUTDIR, "all_reduced_data")
for dir_path in [OUTDIR, TMPDIR]:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
OUTPUT_TXT = os.path.join(TMPDIR, "rating_data.csv")
OUTPUT_TXT_PER_STIM = os.path.join(TMPDIR, "rating_per_stimulus_data.csv")

with open(OUTPUT_TXT, "w") as f:
    header = ["ppname"]
    for condition in CONDITIONS:
        for rating_type in RATING_TYPES:
            header.append("{}_{}".format(condition, rating_type))
    f.write(",".join(header))

with open(OUTPUT_TXT_PER_STIM, "w") as f:
    header = ["ppname"]
    for condition in CONDITIONS:
        for rating_type in RATING_TYPES:
            for i in range(15):
                header.append("{}_{}_{}".format(condition, rating_type, i+1))
    f.write(",".join(header))


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
    if ("_ratings" in name) and (ext == ".txt"):
        n_participants += 1
        ppnames.append(name.replace("_ratings", ""))

# Go through all detected participants.
for ppi, ppname in enumerate(ppnames):
    
    # Load the data.
    fpath = os.path.join(DATADIR, "{}_ratings.txt".format(ppname))
    raw = numpy.loadtxt(fpath, delimiter="\t", skiprows=1, dtype=str)
    # Rename for convenience.
    stim = raw[:,0]
    qtype = raw[:,1]
    resp = raw[:,2].astype(float)
    
    # Split into conditions.
    conditions = []
    for fname in stim:
        name, ext = os.path.splitext(fname)
        condition, stim_nr = name.split("_")
        conditions.append(condition)
    conditions = numpy.array(conditions)
    
    # Compute averages per condition for each rating type.
    with open(OUTPUT_TXT, "a") as f:
        line = [ppname]
        for condition in CONDITIONS:
            for rating_type in RATING_TYPES:
                sel = (conditions == condition) & (qtype == rating_type)
                line.append(numpy.nanmean(resp[sel]))
        f.write("\n" + ",".join(map(str, line)))

    # Compute averages per condition for each rating type.
    with open(OUTPUT_TXT_PER_STIM, "a") as f:
        line = [ppname]
        for condition in CONDITIONS:
            for rating_type in RATING_TYPES:
                for i in range(15):
                    stim_name = "{}_{}.png".format(condition, str(i+1).rjust(2,"0"))
                    sel = (stim == stim_name) & (qtype == rating_type)
                    line.append(resp[sel][0])
        f.write("\n" + ",".join(map(str, line)))

