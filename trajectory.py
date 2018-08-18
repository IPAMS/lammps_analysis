# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import itertools

def read_trajectory(filename, frames_to_read=None):
	"""
	Reads and parses a LAMMPS ASCII trajectory file

	Currently this method assumes an invariant number of particles in the trajectory

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
	with open(filename) as fh:
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
				lines = itertools.islice(fh, n_atoms)
				dat = np.genfromtxt(lines)
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

	:param trajectory: A LAMMPS trajectory
	:type trajectory: xarray with a LAMMPS trajectory
	:param timestep: The timestep of the selected frame
	:type timestep: int
	:param species: A list of species type id's which are selected
	:type species: list of int
	:return: xarray with the filtered trajectory frame
	:rtype: xarray
	"""
	selected_ts = trajectory[timestep]
	frame = selected_ts[np.in1d(selected_ts.loc[:, 'type'], species)]
	return frame


def filter_species_trajectory(trajectory, species):
	"""
	Filters out a trajectory of selected species from a LAMMPS trajectory.

	The returned frames are sorted according to id's, so the indices of the individual particles
	do not change over time.

	:param trajectory: A LAMMPS trajectory
	:type trajectory: xarray with a LAMMPS trajectory
	:param species: A list of species type id's which are selected
	:type species: list of int
	:return: xarray with the filtered trajectory
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