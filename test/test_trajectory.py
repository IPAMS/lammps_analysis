import unittest
import LammpsAnalysis.trajectory as ltra
import os


class Test_trajectory(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.data_base_path = os.path.join('test', 'testfiles')
		uncompressed_trajectory_path = os.path.join(cls.data_base_path, 'trajectories', 'test_trajectory_short.lammpstrj')
		cls.uncompressed_trajectory = ltra.read_trajectory(uncompressed_trajectory_path)

	def test_uncompressed_trajectory_import(self):
		self.assertEqual(self.uncompressed_trajectory.sizes['time'], int(4))
		self.assertEqual(self.uncompressed_trajectory.sizes['particles'], int(11021))

	def test_compressed_trajectory_import(self):
		compressed_trajectory_path = os.path.join(self.data_base_path, 'trajectories', 'test_trajectory_short.lammpstrj.gz')
		compressed_trajectory = ltra.read_trajectory(compressed_trajectory_path)
		self.assertEqual(compressed_trajectory.sizes['time'], int(4))
		self.assertEqual(compressed_trajectory.sizes['particles'], int(11021))

	def test_species_frame_selection(self):
		fr = ltra.filter_species_frame(self.uncompressed_trajectory, 0, [1, 2])
		self.assertTrue('time' not in fr.dims)
		self.assertEqual(fr.sizes['particles'], int(3000))
		t_atom = fr[10,:]
		self.assertEqual(t_atom.loc['type'], 1)
		self.assertEqual(t_atom.loc['id'], 647)
		self.assertAlmostEqual(t_atom.loc['x'].data, 335.252)
		self.assertAlmostEqual(t_atom.loc['y'].data, 165.069)
		self.assertAlmostEqual(t_atom.loc['z'].data, 131.166)
		self.assertAlmostEqual(t_atom.loc['c_ke_all'].data, 1.66836)

	def test_species_trajectory_selection(self):
		tra = ltra.filter_species_trajectory(self.uncompressed_trajectory, [1, 2])
		self.assertEqual(tra.sizes['time'], int(4))
		self.assertEqual(tra.sizes['particles'], int(3000))



