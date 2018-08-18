# -*- coding: utf-8 -*-

import numpy as np
import re


def parse_thermo_log(filename):
	"""
	Parses the "thermo" part of a LAMMPS log.
	The structure of the "thermo" table in a LAMMPS log parsed and its structure
	and values are returned in a tuple

	:param filename: the name of the logfile to parse
	:type filename: str
	:return: a tuple with the (headers of the columns, the times of the timesteps
	and the values of the thermo table
	:rtype: tuple: (list of str, np-array of float, 2 dimensional np-array with values)
	"""
	pt_number_line = re.compile(r"""\s+(?:\-?\d+\.?\d*\s+){10}\n""")
	pt_header_line = re.compile(r"""(?:\w+ ){10}\n""")
	timesteps = []
	values = []
	with open(filename) as fh:
		for line in fh:
			match_number = pt_number_line.match(line)
			if match_number:
				vals = line.split()
				timesteps.append(int(vals[0]))
				values.append([float(vals[i]) for i in range(1, len(vals))])
			else:
				match_header = pt_header_line.match(line)
				if match_header:
					headers = line.split()

	timesteps = np.array(timesteps)
	values = np.array(values)

	return (headers[1:], timesteps, values)

