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
		test_trajectory_path = pt.join(data_base_path, 'trajectories', 'test_trajectory_short.lammpstrj')

		cls.test_result_path = pt.join('test', 'testresults')

		cls.trajectory = ltra.read_trajectory(test_trajectory_path)
		cls.trajectory_water = ltra.filter_species_trajectory(cls.trajectory, [2, 3])

	def test_energy_histogram(self):
		frame = self.trajectory_water[3]
		lvisu.plot_histogram_energy(frame, bins=100)
		plt.savefig(pt.join(self.test_result_path, 'test_test.pdf'), format='pdf')

	def test_energy_timeseries(self):
		lvisu.plot_energies_timeseries(self.trajectory_water,mode='average')
		plt.savefig(pt.join(self.test_result_path, 'test_energy_timeseries_average.pdf'), format='pdf')

		plt.figure()
		lvisu.plot_energies_timeseries(self.trajectory_water,mode='sum')
		plt.savefig(pt.join(self.test_result_path, 'test_energy_timeseries_sum.pdf'), format='pdf')

	def test_radial_density(self):
		tra = self.trajectory
		tra_oxygen = ltra.filter_species_trajectory(tra, [2])
		tra_hydrogen = ltra.filter_species_trajectory(tra, [3])
		tra_ammonium = ltra.filter_species_trajectory(tra, [350])
		tra_methanole = ltra.filter_species_trajectory(tra, [100, 101, 102, 99])

		trajectories = [(tra_hydrogen, 'Hydrogen', True),
		                (tra_oxygen, 'Oxygen', True),
		                (tra_methanole, 'Methanole', True),
		                (tra_ammonium, 'Ammonium', True)]

		# radial density plot of whole averaged trajectory:
		lvisu.plot_radial_density(trajectories, bins=np.linspace(0, 100, 100))
		plt.savefig(pt.join(self.test_result_path, 'test_radial_density.pdf'), format='pdf')
