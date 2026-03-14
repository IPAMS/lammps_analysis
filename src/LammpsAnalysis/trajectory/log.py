# -*- coding: utf-8 -*-

import numpy as np
import re


def parse_run_logs(filename):
	"""
	Parses the run information in a LAMMPS log.

	LAMMPS Log files can contain multiple runs, every run starts with "run ntimesteps",
	some initalization information and lines written to the logfile by the "thermo" fix of LAMMPS

	Currently the thermodynamic data of the, possibly multiple, runs exported by LAMMPS is parsed
	and returned as list with the data from the individual runs

	:param filename: the name of the logfile to parse
	:type filename: str
	:return: a list, one entry per run, each run is a dictionary with the headers of the columns, the times of the timesteps
	and the values of the data exported by the thermo_style fix
	:rtype: list: (list of dict with str, np-array of float, 2 dimensional np-array with values each)
	"""
	pt_header_line = re.compile(r"""Step (?:\w+ )*\n""")

	result = []
	with open(filename) as fh:

		n_params = -1  # number of parameters in the thermo_style of the current run, -1 if no run was seen yet
		headers = None # list of strings: the names of the individual colums of the thermo_style
		pt_number_line = None
		timesteps = None
		values = None
		flag_check_header = False

		for line in fh:

			if n_params != -1:
				match_number = pt_number_line.match(line)
				if match_number:
					vals = line.split()
					timesteps.append(int(vals[0]))
					values.append([float(vals[i]) for i in range(1, len(vals))])

				else:
					flag_check_header = True
			else:
				flag_check_header = True

			if flag_check_header:
				flag_check_header = False
				match_header = pt_header_line.match(line)

				if match_header: # a new thermo_style header was found: prepare the parsing of a new run
					if n_params != -1: #this is not the first run we have found, store the found data
						result.append({
							"headers":headers,
							"timesteps":np.array(timesteps),
							"values":np.array(values)})

					#now prepare the new run:
					headers = line.split()[1:]
					n_params = len(headers)

					# regex to parse the numbers of the thermo_style definition
					pt_number_line = re.compile(r"""\s+(?:\-?\d+\.?\d*(?:e[+-]\d+)?\s+){""" + str(n_params+1) + """}\n""") #+1 for steps column

					timesteps = []
					values = []

		#if the file is parsed to the end and there is a run open, store the found data:
		if n_params != -1:
			result.append({
				"headers": headers,
				"timesteps": np.array(timesteps),
				"values": np.array(values)})

	return (result)
