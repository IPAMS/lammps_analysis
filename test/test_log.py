# -*- coding: utf-8 -*-

import unittest
import lammps_analysis.log as llog

class Test_log(unittest.TestCase):

	def test_parse_basic_log(self):
		runs = llog.parse_run_logs('logs/log.lammps')

		self.assertEqual(len(runs),3)

		self.assertEqual(runs[0]['headers'],['Temp','E_pair','E_mol','TotEng','Press'])

		self.assertEqual(runs[1]['headers'],
		                 ['TotEng', 'KinEng', 'Density', 'Temp', 'Press', 'Volume', 'pot_drop', 'pot_nitr', 'kin_drop'])
		self.assertEqual(runs[1]['timesteps'][-2], 10000)

		self.assertEqual(runs[2]['headers'],
		                 ['TotEng', 'KinEng', 'Density', 'Temp', 'Press', 'Volume', 'pot_drop', 'pot_nitr', 'kin_drop'])
		self.assertEqual(runs[2]['timesteps'][-2],100000)
		self.assertAlmostEqual(runs[2]['values'][1,1], 8146.8869)


	def test_parse_bigcut_log(self):
		runs = llog.parse_run_logs('logs/log_counterion_bigcut.lammps')

		self.assertEqual(len(runs),2)

		self.assertEqual(runs[1]['timesteps'][0], 1000)
		self.assertEqual(runs[1]['timesteps'][-2], 100900)

		self.assertAlmostEqual(runs[1]['values'][3, 3], 8.5079897e-05)


	def test_parse_bigcut_stripped_log(self):
		runs = llog.parse_run_logs('logs/log_counterion_bigcut_stripped.lammps')

		self.assertEqual(len(runs),1)
		self.assertEqual(runs[0]['headers'], ['Temp', 'E_pair', 'E_mol', 'TotEng', 'Press'])
		self.assertEqual(runs[0]['timesteps'][0], 0)
		self.assertEqual(runs[0]['timesteps'][1], 1000)
