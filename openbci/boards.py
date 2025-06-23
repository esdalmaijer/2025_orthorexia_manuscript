#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, \
    BrainFlowPresets
from brainflow.data_filter import DataFilter


class OpenBCI:
    
    def __init__(self, board, ip_address="", ip_port=0, mac_address="", \
        serial_port="", timeout=0, log_file="default.tsv"):
        
        # Set logger (to print info to stderr, not for data logging).
        BoardShim.enable_dev_board_logger()
        
        # Initialise the parameters.
        params = BrainFlowInputParams()
        params.file = ""
        params.ip_address = ip_address
        params.ip_port = ip_port
        params.ip_protocol = 0
        params.mac_address = mac_address
        params.master_board = BoardIds.NO_BOARD
        params.other_info = ""
        params.preset = BrainFlowPresets.DEFAULT_PRESET
        params.serial_number = ""
        params.serial_port = serial_port
        params.timeout = timeout
        
        # Get the board ID, and make sure required inputs are set.
        if board == "ganglion":
            board_id = BoardIds.GANGLION_BOARD
            if params.timeout == 0:
                params.timeout = 15
            if params.serial_port == "":
                if sys.platform == "linux":
                    params.serial_port = "/dev/ttyACM0"
                elif sys.platform == "win32" or sys.platform == "cygwin":
                    params.serial_port = "COM3"
                else:
                    raise Exception("To interface with a Ganglion board, " + \
                        "we need to know what serial port it is connected " + \
                        "to. Your system seems to be macOS, for which we " + \
                        "don't have a default value to try. Please check " + \
                        "the OpenBCI website for info, and then pass the " + \
                        "serial port name as a string using the " + \
                        "serial_port keyword argument.")
                print("WARNING! Ganglion board requires serial_port to be" + \
                    "set. Attempting with default value of '{}'".format( \
                    params.serial_port) + " (but this might not be right!)")

        elif board == "cyton":
            board_id = BoardIds.CYTON_BOARD
            if params.serial_port == "":
                if sys.platform == "linux":
                    params.serial_port = "/dev/ttyUSB0"
                elif sys.platform == "win32" or sys.platform == "cygwin":
                    params.serial_port = "COM3"
                else:
                    raise Exception("To interface with a Cyton board, " + \
                        "we need to know what serial port it is connected " + \
                        "to. Your system seems to be macOS, for which we " + \
                        "don't have a default value to try. Please check " + \
                        "the OpenBCI website for info, and then pass the " + \
                        "serial port name as a string using the " + \
                        "serial_port keyword argument.")
                print("WARNING! Cyton board requires serial_port to be" + \
                    "set. Attempting with default value of '{}'".format( \
                    params.serial_port) + " (but this might not be right!)")

        elif board == "cyton-daisy":
            board_id = BoardIds.CYTON_DAISY_BOARD
            if params.serial_port == "":
                if sys.platform == "linux":
                    params.serial_port = "/dev/ttyUSB0"
                elif sys.platform == "win32" or sys.platform == "cygwin":
                    params.serial_port = "COM3"
                else:
                    raise Exception("To interface with a Cyton board, " + \
                        "we need to know what serial port it is connected " + \
                        "to. Your system seems to be macOS, for which we " + \
                        "don't have a default value to try. Please check " + \
                        "the OpenBCI website for info, and then pass the " + \
                        "serial port name as a string using the " + \
                        "serial_port keyword argument.")
                print("WARNING! Cyton board requires serial_port to be" + \
                    "set. Attempting with default value of '{}'".format( \
                    params.serial_port) + " (but this might not be right!)")

        elif board == "ganglion_wifi":
            board_id = BoardIds.GANGLION_WIFI_BOARD
            if params.ip_address == "":
                params.ip_address = "192.168.4.1"
                print("WARNING! Wifi boards require ip_address to be" + \
                    " set. Using default '{}';".format(params.ip_address) + \
                    " THIS IS A GUESS AND MIGHT NOT BE CORRECT!")
            if params.ip_port == 0:
                params.ip_address = 6677
                print("WARNING! Wifi boards require ip_port to be" + \
                    " set. Using default '{}'".format(params.ip_port) + \
                    " THIS IS A GUESS AND MIGHT NOT BE CORRECT!")

        elif board == "cyton_wifi":
            board_id = BoardIds.CYTON_WIFI_BOARD
            if params.ip_address == "":
                params.ip_address = "192.168.4.1"
                print("WARNING! Wifi boards require ip_address to be" + \
                    " set. Using default '{}';".format(params.ip_address) + \
                    " THIS IS A GUESS AND MIGHT NOT BE CORRECT!")
            if params.ip_port == 0:
                params.ip_address = 6677
                print("WARNING! Wifi boards require ip_port to be" + \
                    " set. Using default '{}'".format(params.ip_port) + \
                    " THIS IS A GUESS AND MIGHT NOT BE CORRECT!")

        elif board == "cyton-daisy_wifi":
            board_id = BoardIds.CYTON_DAISY_WIFI_BOARD
            if params.ip_address == "":
                params.ip_address = "192.168.4.1"
                print("WARNING! Wifi boards require ip_address to be" + \
                    " set. Using default '{}';".format(params.ip_address) + \
                    " THIS IS A GUESS AND MIGHT NOT BE CORRECT!")
            if params.ip_port == 0:
                params.ip_address = 6677
                print("WARNING! Wifi boards require ip_port to be" + \
                    " set. Using default '{}'".format(params.ip_port) + \
                    " THIS IS A GUESS AND MIGHT NOT BE CORRECT!")

        else:
            raise Exception("Board '{}' not recognised.".format(board))

        self._board = BoardShim(board_id, params)
        self._board.prepare_session()
        
        # Set data file name.
        self._log_file = log_file
        # Write a header to the log file.
        self._write_header()
        # Set up a streamer to stream data to the log file.
        self._board.add_streamer("file://{}:a".format(self._log_file), \
            BrainFlowPresets.DEFAULT_PRESET)
        
    
    def _get_data(self):
        
        data = self._board.get_board_data()
        
        return data
    
    def _write_data(self, data=None):
        
        # If no data was passed, grab the currently available data.
        if data is None:
            data = self._get_data()
        # Append the data to log file.
        DataFilter.write_file(data, self._log_file, 'a')
    
    def _write_header(self):

        # Get the board description, which contains channel indices (to make
        # sense of the data rows/columns).
        descr = self._board.get_board_descr(self._board.get_board_id())
        # Unpack the string that is the EEG channel names (why?!) for Cyton.
        if "eeg_names" in descr.keys():
            if type(descr["eeg_names"]) == str:
                descr["eeg_names"] = descr["eeg_names"].split(",")
        # Create an empty header.
        header = [None] * descr["num_rows"]
        # Set channel names in the header.
        for channel_type in ["package_num", "eeg", "accel", "other", \
            "analog", "resistance", "timestamp", "marker"]:
            # Single channel names.
            if channel_type in ["package_num", "timestamp", "marker"]:
                key = "{}_channel".format(channel_type)
                single_channel = True
            # Grouped channel names.
            else:
                key = "{}_channels".format(channel_type)
                single_channel = False
            # Skip channels that do not apply to this board.
            if key not in descr.keys():
                continue
            # Get the channel indices (either list or int type)
            channel_indices = descr[key]
            if single_channel:
                channel_indices = [channel_indices]
            # Loop through all indices for this channel type to set their
            # names in the header.
            for i, hi in enumerate(channel_indices):
                header[hi] = "{}".format(channel_type)
                if len(channel_indices) > 1:
                    header[hi] += "_{}".format(i)
                if (channel_type == "eeg") and ("eeg_names" in descr.keys()):
                    header[hi] += "_{}".format(descr["eeg_names"][i])
        
        with open(self._log_file, "w") as f:
            f.write("\t".join(header) + "\n")
    
    def preview_data(self, n_packages):
        
        # This gets data, but does not remove it from the internal buffer.
        # (Not entirely sure whether it remains available after the stream
        # got to it...)
        self._board.get_current_board_data(n_packages)
    
    def start_stream(self):
        
        self._board.start_stream()
        self._streaming = True
    
    def stop_stream(self):
        
        self._board.stop_stream()
        self._streaming = False
    
    def trigger(self, trigger):
        
        self._board.insert_marker(trigger)
    
    def close(self):
        
        if self._streaming:
            self.stop_stream()
        self._board.release_session()

