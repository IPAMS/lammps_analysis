import unittest
import LammpsAnalysis.trajectory.trajectory as ltra
import LammpsAnalysis.trajectory.analysis as lana
import numpy as np
import os


class Test_analysis(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		data_base_path = os.path.join('test', 'testfiles')
		test_trajectory_path = os.path.join(data_base_path, 'trajectories', 'new_format_test_trajectory.lammpstrj')

		cls.trajectory = ltra.read_trajectory(test_trajectory_path)
		cls.trajectory_water = ltra.filter_species_trajectory(cls.trajectory, [63, 64])
		n_particles = 5
		cls.test_frame = cls.trajectory_water[0].loc[:n_particles,:].copy(deep=True)
		for i in range(n_particles):
			cls.test_frame.loc[i,['x','y','z']] = [i*0.1,i*1,i*10]


	def test_filter_radius(self):
		tra_5 = lana.filter_radius(self.trajectory_water, 5)
		tra_40 = lana.filter_radius(self.trajectory_water, 40)

		self.assertEqual(tra_5[0].sizes['particles'],26)
		self.assertEqual(tra_40[0].sizes['particles'], 1320)

	def test_filter_hemisphere(self):
		tra_x = lana.filter_hemisphere(self.trajectory_water, 'x')
		tra_y = lana.filter_hemisphere(self.trajectory_water, 'y')
		tra_z = lana.filter_hemisphere(self.trajectory_water, 'z')

		self.assertEqual(tra_x[0][0].sizes['particles'], 740)
		self.assertEqual(tra_x[1][0].sizes['particles'], 580)

		self.assertEqual(tra_y[0][0].sizes['particles'], 625)
		self.assertEqual(tra_y[1][0].sizes['particles'], 695)

		self.assertEqual(tra_z[0][0].sizes['particles'], 673)
		self.assertEqual(tra_z[1][0].sizes['particles'], 647)

		center = lana.calculate_geometric_center(self.trajectory_water[0])

		cx = center[0].values
		for part in tra_x[0][0]:
			self.assertTrue(part.loc['x'].values < cx)
			self.assertFalse(part.loc['x'].values > cx)

		for part in tra_x[1][0]:
			self.assertTrue(part.loc['x'].values > cx)
			self.assertFalse(part.loc['x'].values < cx)

		cy = center[1].values
		for part in tra_y[0][0]:
			self.assertTrue(part.loc['y'].values < cy)
			self.assertFalse(part.loc['y'].values > cy)

		for part in tra_y[1][0]:
			self.assertTrue(part.loc['y'].values > cy)
			self.assertFalse(part.loc['y'].values < cy)

		cz = center[2].values
		for part in tra_z[0][0]:
			self.assertTrue(part.loc['z'].values < cz)
			self.assertFalse(part.loc['z'].values > cz)

		for part in tra_z[1][0]:
			self.assertTrue(part.loc['z'].values > cz)
			self.assertFalse(part.loc['z'].values < cz)



	def test_calc_geometric_center(self):
		center = lana.calculate_geometric_center(self.test_frame)
		self.assertAlmostEqual(center[0], 0.2)
		self.assertAlmostEqual(center[1], 2.0)
		self.assertAlmostEqual(center[2], 20.0)

	def test_radii_around_geometric_center(self):
		preset_center = [1.0,2.0,3.0]
		radii,center = lana.radii_around_geometric_center(self.trajectory_water,-1,center=preset_center)
		n_test = 10
		for i in range(n_test):
			x,y,z = self.trajectory_water[-1].loc[i,['x','y','z']]
			r = np.sqrt(
				(x - preset_center[0])**2.0 +
				(y - preset_center[1]) ** 2.0 +
				(z - preset_center[2]) ** 2.0)
			self.assertAlmostEqual(r,radii[i])

