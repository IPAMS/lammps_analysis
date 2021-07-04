# -*- coding: utf-8 -*-

import numpy as np


# advanced filter methods:

def filter_radius(trajectory, radius):
	"""
	Filters out a trajectory in a sphere around the geometric center of the particles.
	The trajectory is returned as a list of single frames because the number of particles in the
	selected sphere can change over time

	:param trajectory: An input trajectory
	:type trajectory: xarray with a read trajectory
	:param radius: the radius around the geom
	:type radius: float
	:return: the filtered trajectory
	:rtype: a list of xarray with the individual filtered frames
	"""
	n_frames = trajectory.sizes['time']
	buf = []
	for timestep in range(n_frames):
		ts_radii, center = radii_around_geometric_center(trajectory, timestep)

		frame = trajectory[timestep].where(ts_radii < radius,drop=True)
		buf.append(frame)
	return (buf)


# low level analysis methods:

def calculate_geometric_center(frame):
	"""
	Calculates the geometrc center of a given timeframe of a trajectory

	:param frame: a timeframe of a LAMMPS trajectory
	:type frame: xarray with a single timeframe
	:return: the geometric center (x,y,z coordinates of the geometric center)
	:rtype: xarray
	"""
	mean_pos = np.mean(frame.loc[:,['x','y','z']],axis=0)
	return(mean_pos)


def radii_around_geometric_center(trajectory, timestep, center=None):
	"""
	Calculates the radii of the individual particles around a geometric center
	in a selected time frame of a LAMPPS trajectory.
	Usually the center to calculate the radii for is the goemetric center of the
	particle cloud in the LAMMPS trajectory.

	:param trajectory: a LAMMPS trajectory
	:type trajectory: xarray with an imported LAMMPS trajectory
	:param timestep: index of the timestep to calculate the radii for
	:type timestep: int
	:param center: optional: center of the radius calculcation, if not set the
	center is the geometric center of the particle cloud in the trajectory
	:type center: array with the x,y,z positions of the center to calculate the radii for
	:return: the calculated radii and the geometric center of the particle cloud
	"""
	ts_frame = trajectory[timestep]
	coords = ts_frame.loc[:, ['x', 'y', 'z']]
	if center is None:
		center = calculate_geometric_center(ts_frame)

	ts_radii = np.sqrt(
		(coords[:, 0] - center[0]) ** 2.0 +
		(coords[:, 1] - center[1]) ** 2.0 +
		(coords[:, 2] - center[2]) ** 2.0)

	return (ts_radii, center)


# util methods:

def normalize(vec):
	"""
	Normalizes a vector, so that all values in the vector are between 0 and 1
	:param vec: the input vector
	:type vec: array
	:return: the normalized vector
	"""
	min_vec = np.min(vec)
	max_vec = np.max(vec)
	n_vec = (vec - min_vec) / (max_vec - min_vec)

	return (n_vec)

