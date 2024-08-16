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
import LammpsAnalysis.cluster.visualization as cl_vis
import LammpsAnalysis.cluster.cluster as cl
import matplotlib.pyplot as plt
import seaborn as sns


### Basic analysis functions

def cluster_count(trajectory):
    frame_counts = []
    for frame in trajectory: 
        count = cl_analysis.filter_clusters_unique_frame(frame.to_pandas())
        frame_counts.append(np.size(count))
    return np.asarray(frame_counts)

def collision_point(trajectory, wall_type):
    kes = cl_analysis.generate_droplet_kinetic_energy_timeseries(trajectory, wall_type)
    infls = cl_analysis.inflection_points(kes)
    return infls[0]

def fragment_spectra(trajectory, frame, limit = 300):
    clusters = cl_analysis.filter_clusters_atom_composition(trajectory, frame, limit)
    return clusters 

def scattering_angles(trajectory, frame, wall_type, wall_vector=np.array([1, 0]), limit=300):
    angles = cl_vis.scattering_angles(trajectory, frame, wall_type, limit, wall_vector)
    return angles

### Expression of specific evaluation functions

def sum_results_cluster_count(input_data, result_data, index, file_res_counter):
    result = cluster_count(input_data)
    if np.any(result_data[index,file_res_counter]):
        result_data[index,file_res_counter] = result_data[index,file_res_counter] + result
    else:
        result_data[index,file_res_counter] = result

def sum_results_collision_point_walltype(wall_type):

    def sum_results_collision_point(input_data, result_data, index, file_res_counter):
        nonlocal wall_type
        result = collision_point(input_data, wall_type)
        if np.any(result_data[index,file_res_counter]):
            result_data[index,file_res_counter] = result_data[index,file_res_counter] + result
        else:
            result_data[index,file_res_counter] = result

    return sum_results_collision_point

def sum_results_scattering_angles_timstep_walltype(frame, wall_type):
    
    def sum_results_scattering_angles(input_data, result_data, index):
        nonlocal frame
        nonlocal wall_type
        result = scattering_angles(input_data, frame, wall_type)
        if np.any(result_data[index,0]):
            result_data[index,0] = np.concatenate((result_data[index,0], result), axis=0)
        else:
            result_data[index,0] = result

    return sum_results_scattering_angles

def sum_results_fragment_spectra_timstep(frame):
    
    def sum_results_fragment_spectra(input_data, result_data, index):
        nonlocal frame
        result = fragment_spectra(input_data, frame)
        if np.any(result_data[index,0]):
            result_data[index,0] = pd.concat([result_data[index,0], result]).groupby('cluster')['occurence'].sum().reset_index()
        else:
            result_data[index,0] = result

    return sum_results_fragment_spectra


### Post-Functions 

def eval_cluster_mean(index):

    def cluster_mean(data):
        nonlocal index
        for variation_index, reprod_sum in enumerate(data[index,:]):
            stddev = np.std(reprod_sum)
            count = np.mean(reprod_sum)
            data[index, variation_index] = [count, stddev]
    return cluster_mean

def eval_cluster_mean_after_collision(index_cluster, index_collision):

    def cluster_mean(data):
        nonlocal index_cluster
        nonlocal index_collision
        
        collision_point = int(np.rint(np.mean(data[index_collision,:])))
        for variation_index, reprod_sum in enumerate(data[index_cluster,:]):
            stddev = np.std(reprod_sum[collision_point:])
            count = np.mean(reprod_sum[collision_point:])
            data[index_cluster, variation_index] = [count, stddev]
    return cluster_mean


### Accumelating base function 

def accumelate_observables(filenames, functions, post_functions, frames_to_read=None):
    function_count = len(functions)
    function_results = np.empty((function_count,1), dtype=object)

    for counter, file in enumerate(filenames): 
        data = cl.read_cluster_data(file, frames_to_read)
        for index, function in enumerate(functions):
            function(data, function_results, index)

    for postindex, postfunction in enumerate(post_functions):           
        postfunction(function_results)
    
    return function_results

### Averaging base function 

def average_observables(filenames, functions, post_functions, reproduction_count, frames_to_read=None):
    function_count = len(functions)
    file_count = len(filenames)
    function_results = np.empty((function_count,file_count//reproduction_count), dtype=object)

    repro_counter = 0
    file_res_counter = 0
    for counter, file in enumerate(filenames): 
        data = cl.read_cluster_data(file, frames_to_read)
        for index, function in enumerate(functions):
            function(data, function_results, index, file_res_counter)
        repro_counter = repro_counter+1 

        if repro_counter >= reproduction_count:
                repro_counter = 0
                for id, element in enumerate(function_results[:, file_res_counter]):
                    function_results[id, file_res_counter] = element/reproduction_count
                file_res_counter = file_res_counter + 1
    
    for postindex, postfunction in enumerate(post_functions):           
        postfunction(function_results)

    return function_results

### Plotting functions

def plot_cluster_count(data, index, voltages):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts = [item[0] for item in data[index, :]]
    stdev = [item[1] for item in data[index, :]]
    lineObj = ax.errorbar(voltages, counts, yerr=stdev, fmt='d', capsize=5)
    ax.set_xlabel('voltage in V')
    ax.set_ylabel('average number of observed cluster')
    plt.show()

    return ax, fig


def plot_collision_point(data, index, voltages):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(voltages, data[index, :], '--x')
    ax.set_xlabel('voltage in V')
    ax.set_ylabel('timestep of collision')
    plt.show()

    return ax, fig

def plot_cluster_composition(data, index):
    fig, ax = plt.subplots(1, 1)
    ax = sns.barplot(data[index, 0].sort_values(by=['occurence'], ascending=False), x="occurence", y="cluster", orient="y")
    ax.set_xlabel("occurence")
    plt.show()

    return ax, fig

def plot_cluster_composition_comparison(data1, index1, data2, index2, comparison_type, classifier1, classifier2):
    clusters1 = data1[index1, 0]
    clusters2 = data2[index2, 0]
    clusters1[comparison_type] = classifier1
    clusters2[comparison_type] = classifier2
    data = pd.concat([clusters1, clusters2])
    fig, ax = plt.subplots(1, 1)
    ax = sns.barplot(data.sort_values(by=['occurence'], ascending=False), x="occurence", y="cluster", orient="y", hue='Wall')
    ax.set_xlabel("occurence")
    plt.show()

    return ax, fig

def plot_scattering_angles(data, index):
    angles = data[index, 0]
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    cl_vis.circular_hist(ax, angles)
    plt.show()

    return ax, fig