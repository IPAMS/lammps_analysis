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


### Expression of specific evaluation functions for accumalting

def sum_results_cluster_count(input_data, result_data, index, frame):
    result = cluster_count(input_data)
    if np.any(result_data[index,0]):
        result_data[index,0] = result_data[index,0] + result
    else:
        result_data[index,0] = result

    return result_data

def sum_results_fragment_spectra_timstep(frame):
    
    def sum_results_fragment_spectra(input_data, result_data, index):
        nonlocal frame
        result = fragment_spectra(input_data, frame)
        if np.any(result_data[index,0]):
            result_data[index,0] = pd.concat([result_data[index,0], result]).groupby('cluster')['occurence'].sum().reset_index()
        else:
            result_data[index,0] = result

    return sum_results_fragment_spectra


### Expression of specific evaluation functions for averaging 


def average_results(input_data, result_data):

    return result_data


### Post-Functions 


def eval_cluster_mean(index):

    def cluster_mean(data):
        nonlocal index
        print(data)
        for variation_index, reprod_sum in enumerate(data[index,:]):
            stddev = np.std(reprod_sum)
            count = np.mean(reprod_sum)
            data[index, variation_index] = [count, stddev]
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

# def evaluate_simulation_run(filenames, functions, post_functions, frames_to_read, reproduction_count, wall_type):
#     function_count = len(functions)
#     file_count = len(filenames)
#     post_functions_count = len(post_functions)
#     function_results = np.empty((function_count,file_count//reproduction_count), dtype=object)

#     repro_counter = 0
#     file_res_counter = 0
#     for counter, file in enumerate(filenames): 
#         data = cl.read_cluster_data(file, frames_to_read)
#         for index, function in enumerate(functions):
#             if function == collision_point:
#                 result = function(data, wall_type)
#             else:
#                 result = function(data)
#             if np.any(function_results[index, file_res_counter]):
#                 function_results[index, file_res_counter] = function_results[index, file_res_counter] + result
#             else:
#                 function_results[index, file_res_counter] = result
                
#             repro_counter = repro_counter+1
#             if repro_counter >= reproduction_count:
#                 repro_counter = 0
#                 for id, element in enumerate(function_results[:, file_res_counter]):
#                     function_results[id, file_res_counter] = element/reproduction_count
#                 file_res_counter = file_res_counter + 1
    
#     for postindex, postfunction in enumerate(post_functions):           
#         postfunction(function_results)

#     return function_results

# def evaluate_timeframe_sums(filenames, functions, post_functions, wall_type, timestep, frames_to_read=None):
#     function_count = len(functions)
#     function_results = np.empty((function_count,1), dtype=object)

#     for counter, file in enumerate(filenames): 
#         data = cl.read_cluster_data(file, frames_to_read)
#         for index, function in enumerate(functions):
           
#             result = function(data, timestep)
#             if np.any(function_results[index,0]):
#                 function_results[index,0] = pd.concat([function_results[index,0], result]).groupby('cluster')['occurence'].sum().reset_index()
#             else:
#                 function_results[index,0] = result   

#     for postindex, postfunction in enumerate(post_functions):           
#         postfunction(function_results)
    
#     return function_results


def plot_cluster_count(data, index):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts = [item[0] for item in data[index, :]]
    stdev = [item[1] for item in data[index, :]]
    lineObj = ax.errorbar([10,20,30,40], counts, yerr=stdev, fmt='x', capsize=5)
    ax.set_xlabel('voltage in V')
    ax.set_ylabel('average number of observed cluster')
    plt.show()

    return ax, fig


def plot_collision_point(data, index):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([10,20,30,40,50,60,70,80,90,100], data[index, :], '--x')
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