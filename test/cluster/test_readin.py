import unittest
import LammpsAnalysis.cluster.cluster as cl
import os


class Test_trajectory(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.data_base_path = os.path.join('test', 'testfiles')
		uncompressed_trajectory_path = os.path.join(cls.data_base_path, 'cluster', 'cluster_data_trajectory')
		cls.uncompressed_trajectory = cl.read_cluster_data(uncompressed_trajectory_path, 3)

	def test_uncompressed_trajectory_import(self):
		self.assertEqual(self.uncompressed_trajectory.sizes['time'], int(1))
		self.assertEqual(self.uncompressed_trajectory.sizes['particles'], int(41977))

	def test_species_frame_selection(self):
		fr = cl.xarray_filter_species_frame(self.uncompressed_trajectory, 0, [907, 700])
		self.assertTrue('time' not in fr.dims)
		self.assertEqual(fr.sizes['particles'], int(39642))
		t_atom = fr[10,:]
		self.assertEqual(t_atom.loc['Type'], 907)
		self.assertEqual(t_atom.loc['ID'], 19)
		self.assertAlmostEqual(t_atom.loc['X'].data, 4.00972)
		self.assertAlmostEqual(t_atom.loc['Y'].data, 206.273)
		self.assertAlmostEqual(t_atom.loc['Z'].data, 202.331)

	def test_species_trajectory_selection(self):
		tra = cl.filter_species_trajectory(self.uncompressed_trajectory, [907, 700])
		self.assertEqual(tra.sizes['time'], int(1))
		self.assertEqual(tra.sizes['particles'], int(39642))