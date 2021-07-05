import LammpsAnalysis as la
import argparse
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('tra_name', help='Filename for trajectory file to read')
args = parser.parse_args()
start = timer()
trj = la.read_trajectory(args.tra_name)
end = timer()
print('Elapsed:', end - start)
print(trj.sizes['particles'])
print(trj.sizes['time'])