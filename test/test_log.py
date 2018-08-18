# -*- coding: utf-8 -*-

import unittest
import lammps_analysis.log as llog


class Test_log(unittest.TestCase):

	def test_parse_log(self):
		headers,ts,vals = llog.parse_thermo_log('log.lammps')
		self.assertEqual(headers[0],'TotEng')
		self.assertEqual(headers[5], 'Volume')
		self.assertEqual(ts[-2],100000)
		self.assertAlmostEqual(vals[3,0], -10203.461)

