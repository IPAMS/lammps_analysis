import math 
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import LammpsAnalysis.cluster.cluster as cl_cluster
from collections import Counter
import pandas as pd
import re
from chemformula import ChemFormula
import LammpsAnalysis.cluster.analysis as cl_analysis
import LammpsAnalysis.cluster.cluster as cl
import matplotlib.pyplot as plt


def cluster_count(trajectory):
    frame_counts = []
    for frame in trajectory: 
        count = cl_analysis.filter_clusters_unique_frame(frame.to_pandas())
        frame_counts.append(np.size(count))
    stddev = np.std(frame_counts)
    count = np.mean(frame_counts)

    return [count, stddev]

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # lineObj = ax.errorbar(voltages, counts, yerr=stddevs, fmt='x', capsize=5)
    # ax.set_xlabel('voltage in V')
    # ax.set_ylabel('average cluster count')
    # plt.show()

def collision_point(trajectory, wall_type):
    kes = cl_analysis.generate_droplet_kinetic_energy_timeseries(trajectory, wall_type)
    infls = cl_analysis.inflection_points(kes)
    return infls[0]

def evaluate_simulation_run(filenames, voltages, functions, frames_to_read, wall_type):
    function_count = len(functions)
    file_count = len(filenames)
    function_results = np.empty((function_count,file_count), dtype=object)
    for counter, file in enumerate(filenames): 
        data = cl.read_cluster_data(file, frames_to_read)
        for index, function in enumerate(functions):
            if function == collision_point:
                result = function(data, wall_type)
            else:
                result = function(data)
            function_results[index, counter] = result
    return function_results


def plot_cluster_count(data, index):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts = [item[0] for item in data[index,:]]
    stdev = [item[1] for item in data[index,:]]
    lineObj = ax.errorbar([10,20,30,40,50,60,70,80,90,100], counts, yerr=stdev, fmt='x', capsize=5)
    ax.set_xlabel('voltage in V')
    ax.set_ylabel('average cluster count')
    plt.show()


def plot_collision_point(data, index):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([10,20,30,40,50,60,70,80,90,100], data[index,:], '--x')
    ax.set_xlabel('voltage in V')
    ax.set_ylabel('timestep of collision')
    plt.show()


