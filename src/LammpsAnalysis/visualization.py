# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
import xarray as xr
from matplotlib import cm
import ipyvolume.pylab as p3

import LammpsAnalysis.analysis as la


def plot_histogram_energy(frame, bins=100, col_name_pe='c_pe_all', col_name_ke='c_ke_all'):
	"""
	Plots a histogram of the energy distribution in a given uncompressed_trajectory frame.
	The names of the data columns in the uncompressed_trajectory to generate the histogram for
	are variable.

	:param frame: a timeframe of a LAMMPS uncompressed_trajectory
	:type frame: xarray
	:param bins: the bins for the histogram (according to the matplotlib histogram method)
	:type bins: int or numpy.array or list
	:param col_name_pe: data column name for the potential energy
	:type col_name_pe: str
	:param col_name_ke: data column name for the kinetic energy
	:type col_name_ke: str
	"""
	pe = frame.loc[:, col_name_pe]
	ke = frame.loc[:, col_name_ke]
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.hist(pe, bins=bins)
	plt.title("Potential Energy")
	plt.xlabel("Potential Energy  (kcal/mol)")
	plt.ylabel("# particles")

	plt.subplot(1, 2, 2)
	plt.hist(ke, bins=bins)
	plt.title("Kinetic Energy")
	plt.xlabel("Kinetic Energy  (kcal/mol)")
	plt.ylabel("# particles")


def plot_energies_timeseries(trajectory, mode='average', col_name_pe='c_pe_all', col_name_ke='c_ke_all'):
	"""
	Plots a line plot of the average or summed energies in a time series of
	LAMMPS simulation uncompressed_trajectory frames

	:param trajectory: the uncompressed_trajectory / uncompressed_trajectory frame time series
	:type trajectory: list of uncompressed_trajectory frames or xarray of stacked frames with time as
	one dimension
	:param mode: 'average' calculates the averaged energies, 'sum' the sum of energies
	:type mode: str
	:param col_name_pe: data column name for the potential energy
	:type col_name_pe: str
	:param col_name_ke: data column name for the kinetic energy
	:type col_name_ke: str
	:return: none
	"""

	if mode == 'average':
		calc_func = np.mean
		title_prefix = 'Averaged '
	elif mode == 'sum':
		calc_func = np.sum
		title_prefix = 'Summed '



	if isinstance(trajectory, list):
		mean_ke = [calc_func(frame.loc[:, col_name_ke]) for frame in trajectory]
		mean_pe = [calc_func(frame.loc[:, col_name_pe]) for frame in trajectory]
	else:
		ke = trajectory.loc[:, :, col_name_ke]
		pe = trajectory.loc[:, :, col_name_pe]
		mean_ke = calc_func(ke, axis=1)
		mean_pe = calc_func(pe, axis=1)

	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.plot(mean_pe)
	plt.title(title_prefix+"Potential Energy")
	plt.ylabel(title_prefix+"Potential Energy  (kcal/mol)")
	plt.xlabel("Time Steps")

	plt.subplot(1, 2, 2)
	plt.plot(mean_ke)
	plt.title(title_prefix+"Kinetic Energy")
	plt.ylabel(title_prefix+"Kinetic Energy  (kcal/mol)")
	plt.xlabel("Time Steps")


def plot_radial_density(trajectories, bins=50, selected_frames='all'):
	"""
	Plots the averaged radial densities for multiple species specific LAMMPS trajectories

	The trajectories parameter consists of a list of individual species specific trajectories
	with their plot configurations:


	:param trajectories: list of trajectories and configurations
	:type trajectories: list
	:param bins: the bins for the histogram (according to the matplotlib histogram method)
	:type bins: int or numpy.array or list
	:param selected_frames: a list of frames to consider in the averaging,
	'all' means all frames are averaged
	:type selected_frames: list of int or 'all'
	:return: None
	"""
	plt.figure(figsize=(15, 5))

	# determine which frames to consider:
	n_frames = trajectories[0][0].sizes['time']
	if selected_frames == 'all':
		frames = np.arange(n_frames)
	else:
		frames = selected_frames

	# prepare a dict to retain calculated radii:
	buf = {tra[1]: [] for tra in trajectories}

	for frame in frames:
		# calculate center for the frame:
		# (append selected coordinates and calculate center for them)
		center_coords = []
		for tra in trajectories:
			if tra[2] == True:
				center_coords.append(tra[0][frame].loc[:, ['x', 'y', 'z']])
		center_coords = xr.concat(center_coords, dim='particles')
		center = np.mean(center_coords, axis=0)

		# calculate radii and store them:
		for tra in trajectories:
			fr_radii, fr_center = la.radii_around_geometric_center(tra[0], frame, center=center)
			buf[tra[1]].append(fr_radii)

	# plot results
	for tra in trajectories:
		radii = xr.concat(buf[tra[1]], dim='time')
		particles = radii.values
		hbins, edges = np.histogram(particles, bins, normed=1)
		left, right = edges[:-1], edges[1:]
		X = np.array([left, right]).T.flatten()
		Y = np.array([hbins, hbins]).T.flatten()

		plt.plot(X, Y, label=tra[1])

	plt.legend()
	plt.xlabel("radial distance (Ängström)")
	plt.ylabel("particle abundance")


## 3d visualization:

def scatter_animation(trajectory, color_param='c_ke_all'):
	"""
	Creates an animated three dimensional scatter plot (with ipyvolume)

	Currently the uncompressed_trajectory has to be a sorted uncompressed_trajectory with invariant number
	of particles in the individual uncompressed_trajectory frames

	:param trajectory: the LAMMPS uncompressed_trajectory to visualize
	:param color_param: the name of the data column to use to colorize the particles
	:type color_param: str
	:return: None
	"""
	x = np.array(trajectory.loc[:, :,'x'])
	y = np.array(trajectory.loc[:, :,'y'])
	z = np.array(trajectory.loc[:, :,'z'])

	cp = np.array(trajectory.loc[:, :, color_param])

	colormap = cm.plasma
	cp_c = colormap(la.normalize(cp))[: ,: ,:3]

	t_min = np.min(np.min(trajectory.loc[: ,: ,['x' ,'y' ,'z']] ,axis=1) ,axis=0)
	t_max = np.max(np.max(trajectory.loc[: ,: ,['x' ,'y' ,'z']] ,axis=1) ,axis=0)

	p3.figure()
	scatterplot = p3.scatter(x ,y ,z ,marker="sphere" ,size=0.2 ,color=cp_c)
	p3.xlim(t_min[0] ,t_max[0])
	p3.ylim(t_min[1] ,t_max[1])
	p3.zlim(t_min[2] ,t_max[2])
	p3.animation_control(scatterplot) # shows controls for animation controls
	p3.show()

