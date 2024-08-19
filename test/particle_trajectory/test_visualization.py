import unittest
import LammpsAnalysis.trajectory.trajectory as ltra
import LammpsAnalysis.trajectory.visualization as lvisu
import matplotlib.pylab as plt
import numpy as np
import os.path as pt


class Test_analysis(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		data_base_path = pt.join('test', 'testfiles')
		test_trajectory_path = pt.join(data_base_path, 'trajectories', 'new_format_test_trajectory.lammpstrj')

		cls.test_result_path = pt.join('test', 'testresults')

		cls.trajectory = ltra.read_trajectory(test_trajectory_path)
		cls.trajectory_water = ltra.filter_species_trajectory(cls.trajectory, [63, 64])

	def test_energy_histogram(self):
		frame = self.trajectory_water[3]
		lvisu.plot_histogram_energy(frame, bins=100)
		plt.savefig(pt.join(self.test_result_path, 'test_test.pdf'), format='pdf')

	def test_energy_timeseries(self):
		fig_avg = lvisu.plot_energies_timeseries(self.trajectory_water,mode='average')
		fig_avg.savefig(pt.join(self.test_result_path, 'test_energy_timeseries_average.pdf'), format='pdf')

		#plt.figure()
		fig_sum = lvisu.plot_energies_timeseries(self.trajectory_water, mode='sum')
		fig_sum.savefig(pt.join(self.test_result_path, 'test_energy_timeseries_sum.pdf'), format='pdf')

	def test_droplet_thermalization_plot(self):
		tra = self.trajectory
		tra_water = ltra.filter_species_trajectory(tra, [63, 64])

		fig_sum_x = lvisu.plot_energies_timeseries(self.trajectory_water, mode='sum', segmentation_mode='center_x')
		fig_sum_x.savefig(pt.join(self.test_result_path, 'test_thermalization_x.pdf'), format='pdf')

		fig_sum_y = lvisu.plot_energies_timeseries(self.trajectory_water, mode='average', segmentation_mode='center_y')
		fig_sum_y.savefig(pt.join(self.test_result_path, 'test_thermalization_y.pdf'), format='pdf')

		fig_sum_z = lvisu.plot_energies_timeseries(self.trajectory_water, mode='sum', segmentation_mode='center_z')
		fig_sum_z.savefig(pt.join(self.test_result_path, 'test_thermalization_z.pdf'), format='pdf')


	def test_radial_density(self):
		tra = self.trajectory
		tra_oxygen = ltra.filter_species_trajectory(tra, [63])
		tra_hydrogen = ltra.filter_species_trajectory(tra, [64])

		trajectories = [(tra_hydrogen, 'Hydrogen', True),
		                (tra_oxygen, 'Oxygen', True)]

		# radial density plot of whole averaged trajectory:
		lvisu.plot_radial_density(trajectories, bins=np.linspace(0, 100, 100))
		plt.savefig(pt.join(self.test_result_path, 'test_radial_density.pdf'), format='pdf')
