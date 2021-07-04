# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import itertools
import io
import os
import gzip


def read_trajectory(filename, frames_to_read=None):
	"""
	Reads and parses a LAMMPS ASCII uncompressed_trajectory file

	Currently this method assumes an invariant number of particles in the uncompressed_trajectory

	:param filename: The filename to read
	:type filename: str
	:param frames_to_read: number of frames to read, if not given all frames in the file are read
	:type frames_to_read: int
	:return: An xarray with the read data
	:rtype: xarray
	"""

	n_frames = 0
	n_lines = 0
	time = 0
	data = []

	# todo: use gzip object if gzip, use file else...
	basename, ext = os.path.splitext(filename)
	if ext == ".gz":
		open_fct = gzip.open
	else:
		open_fct = open

	with open_fct(filename, mode='rt') as fh:
		for line in fh:
			n_lines += 1
			if line == "ITEM: TIMESTEP\n":
				# a new timestep has begun
				n_frames += 1
				time = int(next(fh))
			elif line == "ITEM: NUMBER OF ATOMS\n":
				n_atoms = int(next(fh))
			# print(n_atoms)
			elif "ITEM: ATOMS " in line:
				l_split = line.split()
				parcoords = l_split[2:]

				strio = io.StringIO()
				lines = itertools.islice(fh, n_atoms)
				for line in lines:
					strio.write(line)
				strio.seek(0)
				dat = pd.read_csv(strio, delimiter=' ', header=None).values[:, :-1]

				data.append(
					xr.DataArray(dat, dims=('particles', 'params'), coords={'params': parcoords, 'time': time})
				)

			if frames_to_read and n_frames >= frames_to_read:
				break

	combined_data = xr.concat(data, dim='time')  # ,coords={'params': parcoords,'time':timesteps})
	return combined_data


def filter_species_frame(trajectory, timestep, species):
	"""
	Filters out a single timeframe with selected species
	(The returned frames is not sorted according to id's)

	:param trajectory: A LAMMPS uncompressed_trajectory
	:type trajectory: xarray with a LAMMPS uncompressed_trajectory
	:param timestep: The timestep of the selected frame
	:type timestep: int
	:param species: A list of species type id's which are selected
	:type species: list of int
	:return: xarray with the filtered uncompressed_trajectory frame
	:rtype: xarray
	"""
	selected_ts = trajectory[timestep]
	frame = selected_ts[np.in1d(selected_ts.loc[:, 'type'], species)]
	return frame


def filter_species_trajectory(trajectory, species):
	"""
	Filters out a uncompressed_trajectory of selected species from a LAMMPS uncompressed_trajectory.

	The returned frames are sorted according to id's, so the indices of the individual particles
	do not change over time.

	:param trajectory: A LAMMPS uncompressed_trajectory
	:type trajectory: xarray with a LAMMPS uncompressed_trajectory
	:param species: A list of species type id's which are selected
	:type species: list of int
	:return: xarray with the filtered uncompressed_trajectory
	:rtype: xarray
	"""
	n_frames = trajectory.sizes['time']
	buf = []
	for i in range(n_frames):
		frame_unsorted = filter_species_frame(trajectory, i, species)
		frame = frame_unsorted.sortby(frame_unsorted.loc[:, 'id'])
		buf.append(frame)
	result = xr.concat(buf, dim='time')
	return result