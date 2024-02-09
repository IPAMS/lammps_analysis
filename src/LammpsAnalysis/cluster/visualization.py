import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
import matplotlib.colors
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns
import numpy as np 
import LammpsAnalysis.cluster.analysis as cl_analysis
import pandas as pd

def plot_cluster_count_trajectory(trajectory):
    """
    Plots the cluster count for each frame of the trajectory

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :return: figure axis
    :rtype: mpl axis
    """
    cluster_count = cl_analysis.cluster_count_trajectory(trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cluster_count)
    ax.set_xlabel('timestep')
    ax.set_ylabel('cluster count')
    plt.show()

    return ax


def plot_droplet_kinetic_energy_timeseries(trajectory, wall_type, collision_limit=5):
    """
    Plots the kinetic energy for each frame of the trajectory.
    Additionally, the point of collision between droplet and wall is calculated.

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param wall_type: ID of wall atoms
    :type wall_type: int
    :param collision_limit: number of frames around the collision to plot seperately, defaults to 5
    :type collision_limit: int, optional
    :return: figure axes
    :rtype: mpl axes
    """
    kes = cl_analysis.generate_droplet_kinetic_energy_timeseries(trajectory, wall_type)
    infls = cl_analysis.inflection_points(kes)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kes)
    plt.xlabel("timestep")
    plt.ylabel("kinetic energy of droplet in eV")

    ax2 = plt.axes([.56, .4, .3, .3])
    ax.scatter(x=infls[0], y=kes[infls[0]], color='green', label='Collision point', marker='x')
    ax2.scatter(x=infls[0], y=kes[infls[0]], color='green', label='Collision point', marker='x')
    
    ax2.plot(list(range(infls[0]-collision_limit,infls[0]+collision_limit)), 
                        kes[infls[0]-collision_limit:infls[0]+collision_limit])
    ax2.set_facecolor('white')
    ax.legend()
    mark_inset(ax, ax2, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.show()

    return ax, ax2

def plot_cluster_distribution(trajectory, timestep, savefig=False):    
    """
    Plots distribution of cluster in the xy plane for a given frame

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param timestep: frame number
    :type timestep: int
    :param savefig: flag if plot should be saved, defaults to False
    :type savefig: bool, optional
    :return: figure axis
    :rtype: mpl axis
    """
    coms, masses, clusters = cl_analysis.get_all_cluster_distributions_in_space(trajectory, timestep)
    reldata = pd.DataFrame({'x':coms[:,0], 'y':coms[:,1], 'z':coms[:,2], 'mass':np.round(masses,3), 'cluster':np.round(clusters,0)})

    sc = sns.scatterplot(reldata, x='x', y='y', hue='cluster', size='mass', palette="magma", legend='full', size_norm=matplotlib.colors.LogNorm())

    box = sc.get_position()
    sc.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    sc.set_xlabel('x in Å')
    sc.set_ylabel('y in Å')
    sc.legend(loc='upper left', framealpha=1, bbox_to_anchor=(-0.12, -0.15), ncol=5)

    if(savefig):
        plt.savefig('cluster_distribution.pdf', bbox_inches='tight')

    plt.show()
    return sc

def plot_boxplot_cluster_sizes_trajectory(trajectory, limit = 40, step=10, offset=0):
    """
    Plots boxplot of cluster sizes for specified timesteps. 
    A limit allows to only plot below a given threshold of atoms in a cluster 
    (essentially removing the wall cluster)

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param limit: count limit, defaults to 40
    :type limit: int, optional
    :param step: step size between frames, defaults to 10
    :type step: int, optional
    :param offset: start frame, defaults to 0
    :type offset: int, optional
    :return: figure axis
    :rtype: mpl axis
    """
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    result = pd.DataFrame()
    for i in range(0+offset, trajectory.shape[0], step):
        counts = cl_analysis.count_atoms_clusters_unique_frame(trajectory[i])
        result = pd.concat(axis=0, ignore_index=True, objs=[
            result,
            pd.DataFrame.from_dict({'count':counts, 'timestep':i})
        ])
    
    sc = sns.boxplot(result[result['count'] < limit], x='count', y='timestep', hue='timestep', palette='viridis', orient="h", ax=ax)
    sc.set_xlabel('atom count in clusters')
    sc.set_ylabel('timestep')
    sns.stripplot(result[result['count'] < limit],  x='count', y='timestep', size=4, orient="h", ax=ax, color="#824055", dodge=True, legend=False)
    sns.despine(trim=True, left=True)
    sc.legend(framealpha=1, title='timestep')
    plt.show()
    return sc

def plot_boxplot_cluster_masses_trajectory(trajectory, limit = 200, step=10, offset=0):
    """
    Plots boxplot of cluster masses for specified timesteps. 
    A limit allows to only plot below a given threshold of cluster mass 
    (essentially removing the wall cluster)

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param limit: mass limit, defaults to 200
    :type limit: int, optional
    :param step: step size between frames, defaults to 10
    :type step: int, optional
    :param offset: start frame, defaults to 0
    :type offset: int, optional
    :return: figure axis
    :rtype: mpl axis
    """
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    result = pd.DataFrame()
    for i in range(0+offset, trajectory.shape[0], step):
        masses = cl_analysis.count_mass_clusters_unique_frame(trajectory[i])
        result = pd.concat(axis=0, ignore_index=True, objs=[
            result,
            pd.DataFrame.from_dict({'mass':masses, 'timestep':i})
        ])
    sc = sns.boxplot(result[result['mass'] < limit], x='mass', y='timestep', hue='timestep', palette='viridis', orient="h")
    
    sc.set_xlabel('cluster mass in Da')
    sc.set_ylabel('timestep')
    sns.stripplot(result[result['mass'] < limit],  x='mass', y='timestep', size=4, orient="h", ax=ax, color="#824055", dodge=True, legend=False)
    sns.despine(trim=True, left=True)
    sc.legend(framealpha=1, title='timestep')
    plt.show()
    return sc

def plot_histogram_atoms_cluster(trajectory, timestep):
    """
    Plots histogram of number of atoms in all clusters for a given timestep.
    Additionally, all bins under  a limit are plotted in a smaller window

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param timestep: frame number
    :type timestep: int
    :return: figure axes
    :rtype: mpl axes
    """
    counts = cl_analysis.count_atoms_clusters_unique(trajectory,timestep)

    reldata = pd.DataFrame({'atom count':counts})
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    sc = sns.histplot(reldata, x='atom count', log_scale=True, stat='count', ax=ax1)
    sc.set_ylabel('count #')

    ax2 = plt.axes([.46, .5, .4, .3])
    sc2 = sns.histplot(reldata[reldata < 40], x='atom count', stat='count', discrete=True, ax=ax2)
    ax2.set_facecolor('white')
    sc2.set_ylabel('count #')
    plt.show()
    return sc, sc2

def plot_histogram_mass_cluster(trajectory, timestep):
    """
    Plots histogram of mass of all clusters for a given timestep.
    Additionally, all bins under a limit are plotted in a smaller window

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param timestep: frame number
    :type timestep: int
    :return: figure axes
    :rtype: mpl axes
    """
    masses = cl_analysis.count_mass_clusters_unique(trajectory,timestep)

    reldata = pd.DataFrame({'cluster masses':masses})
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    sc = sns.histplot(reldata, x='cluster masses', log_scale=True, stat='count', ax=ax1)
    sc.set_ylabel('count #')
    sc.set_xlabel('cluster masses in Da')


    ax2 = plt.axes([.46, .5, .4, .3])
    sc2 = sns.histplot(reldata[reldata < 200], x='cluster masses', stat='count', discrete=True, ax=ax2)
    ax2.set_facecolor('white')
    sc2.set_ylabel('count #')
    plt.show()
    return sc, sc2


def plot_distribution_cluster_size_timeseries(trajectory, limit=40, step=10, offset=0):
    """
    Plots cluster size distribution for multiple frames in a single histogram

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param limit: cluster size limit, defaults to 40
    :type limit: int, optional
    :param step: step size between frames, defaults to 10
    :type step: int, optional
    :param offset: start frame, defaults to 0
    :type offset: int, optional
    :return: figure axis
    :rtype: mpl axis
    """
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    result = pd.DataFrame()
    for i in range(0+offset, trajectory.shape[0], step):
        counts = cl_analysis.count_atoms_clusters_unique_frame(trajectory[i])
        result = pd.concat(axis=0, ignore_index=True, objs=[
            result,
            pd.DataFrame.from_dict({'count':counts, 'timestep':i})
        ])
    sc = sns.histplot(result[result['count'] < limit], x='count', hue='timestep', 
                      palette='magma_r', stat='count', ax=ax, legend=True, multiple="dodge", binwidth=1.5, discrete=False)
    sc.set_xlabel('atom count in clusters')
    sc.set_ylabel('count #')
    sns.move_legend(sc, "upper right", ncol=5, frameon=True)
    plt.show()
    return sc


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on axis

    :param ax: axis instance
    :type ax: mpl axis
    :param x: Angles to plot, expected in units of radians
    :type x: list
    :param bins: defines the number of equal-width bins in the range, defaults to 16
    :type bins: int, optional
    :param density: If True plot frequency proportional to area. If False plot frequency
        proportional to radius, defaults to True
    :type density: bool, optional
    :param offset: offset for the location of the 0 direction in units of
        radians, defaults to 0
    :type offset: int, optional
    :param gaps: whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range, defaults to True
    :type gaps: bool, optional
    :return: number of values in each bin, edges of bins, container of individual artists used to create the histogram
    :rtype: list, list, container/polygon
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=True, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    #ax.contourf(bins[:-1], radius, n, 30)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches



def scattering_angles(trajectory, timestep, wall_type, limit, wall_vector=np.array([1, 0])):
    """
    Calculates scattering angles between clusters and wall given by normal vector

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param timestep: frame number
    :type timestep: int
    :param wall_type: ID of wall atoms
    :type wall_type: int
    :param limit: limit for cluster mass to include
    :type limit: float
    :param wall_vector: normal vector of wall/vector to take the angle with, defaults to np.array([1, 0])
    :type wall_vector: list, optional
    :return: scattering angles of all clusters
    :rtype: list
    """
    kes = cl_analysis.generate_droplet_kinetic_energy_timeseries(trajectory, wall_type)
    infls = cl_analysis.inflection_points(kes)
    collision_point = infls[0]
    selected_ts = trajectory[timestep].to_pandas()
    clusters = cl_analysis.filter_clusters_unique_frame(selected_ts)
    vels = []

    for cluster in clusters: 
        vel = cl_analysis.cluster_center_of_velocity_direction(selected_ts, cluster, True)[0:2] # projection on xy
        mass = cl_analysis.cluster_mass(selected_ts, cluster)
        vel_mag = np.linalg.norm(np.array(vel))
        if mass < limit:
            vels.append(vel)
    angles = []

    for vel in vels: 
        if vel[1] < 0: 
            angle = 2*np.pi- np.arccos(np.dot(wall_vector,np.array(vel)) / (np.linalg.norm(wall_vector)*(np.linalg.norm(np.array(vel)))))
        else: 
            angle = np.arccos(np.dot(wall_vector,np.array(vel)) / (np.linalg.norm(wall_vector)*(np.linalg.norm(np.array(vel)))))

        angles.append(angle)

    angles = np.array(angles)
    return angles

def radial_distribution_histogram(trajectory, timestep, wall_type, ax, limit, wall_vector=np.array([1, 0])):
    """
    Computes one histogram of scattering angles for a given timestep of the cluster trajectory.
    This function is used in the animation of scattering angle evolution

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param timestep: frame number
    :type timestep: int
    :param wall_type: ID of wall atoms
    :type wall_type: int
    :param ax: axis to plot to
    :type ax: mpl axis
    :param limit: mass limit on the clusters
    :type limit: float
    :param wall_vector: normal vector of wall/vector to take the angle with, defaults to np.array([1, 0])
    :type wall_vector: list, optional
    """

    angles = scattering_angles(trajectory, timestep, wall_type, limit, wall_vector)
    circular_hist(ax, angles)
    plt.show()

def plot_radial_distribution(trajectory, timestep, wall_type, limit, wall_vector=np.array([1, 0])):
    """
    Plots one histogram of scattering angles for a given timestep of the cluster trajectory

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param timestep: frame number
    :type timestep: int
    :param wall_type: ID of wall atoms
    :type wall_type: int
    :param limit: mass limit on the clusters
    :type limit: float
    :param wall_vector: normal vector of wall/vector to take the angle with, defaults to np.array([1, 0])
    :type wall_vector: list, optional
    :return: figure axis
    :rtype: mpl axis
    """
    
    angles = scattering_angles(trajectory, timestep, wall_type, limit, wall_vector)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    circular_hist(ax, angles)
    plt.show()
    return ax


def animate_radial_distribution(trajectory, wall_type, animation_range, output, limit, wall_vector=np.array([1, 0])):
    """
    Animates scattering angle distrubtion over a given range of frames for the cluster trajectory

    :param trajectory: cluster data trajectory
    :type trajectory: xarray
    :param wall_type: ID of wall atoms
    :type wall_type: int
    :param animation_range: range for animation over the frames
    :type animation_range: list
    :param output: filename to save the animation
    :type output: string
    :param limit: mass limit for the selected clusters
    :type limit: float
    :param wall_vector: normal vector of wall/vector to take the angle with, defaults to np.array([1, 0])
    :type wall_vector: list, optional
    """
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))

    x = range(animation_range[0], animation_range[1])

    def make_frame(i):
        ax.clear()
        radial_distribution_histogram(trajectory, i, wall_type, ax, limit, wall_vector)

    # creating animation
    animation = FuncAnimation(fig,make_frame,frames=x) 
    plt.close()

    from matplotlib.animation import PillowWriter
    # Save the animation as an animated GIF
    animation.save(output+".gif", dpi=400,
             writer=PillowWriter(fps=5))
    

def plot_series_cluster_count(trajectories, voltages, timestep):
    """
    Plots the total cluster count in a given timestep for a series of cluster 
    data trajectories which vary in their "acceleration" voltage

    :param trajectories: cluster data trajectories
    :type trajectories: list
    :param voltages: voltages for each trajectory
    :type voltages: list
    :param timestep: frame number
    :type timestep: int
    :return: figure axis 
    :rtype: mpl axis
    """
    counts = []
    for trajectory in trajectories:
        count = cl_analysis.filter_clusters_unique(trajectory, timestep)
        counts.append(np.size(count))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lineObj = ax.plot(voltages, counts, '--x', label='Timestep: ' + str(timestep))
    ax.legend()
    ax.set_xlabel('voltage in V')
    ax.set_ylabel('average cluster count')
    plt.show()

    return ax

def plot_series_cluster_count_whole_trajectory(trajectories, voltages):
    """
    Plots the mean cluster count and the std. deviation for a series of cluster 
    data trajectories which vary in their "acceleration" voltage

    :param trajectories: cluster data trajectories
    :type trajectories: list
    :param voltages: voltages for each trajectory
    :type voltages: list
    :return: figure axis 
    :rtype: mpl axis
    """
    counts = []
    stddevs = []
    for trajectory in trajectories:
        frame_counts = []
        for frame in trajectory: 
            count = cl_analysis.filter_clusters_unique_frame(frame.to_pandas())
            frame_counts.append(np.size(count))
        stddevs.append(np.std(frame_counts))
        counts.append(np.mean(frame_counts))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lineObj = ax.errorbar(voltages, counts, yerr=stddevs, fmt='x', capsize=5)
    ax.set_xlabel('voltage in V')
    ax.set_ylabel('average cluster count')
    plt.show()

    return ax

def plot_series_collision_points(trajectories, voltages, wall_type):
    """
    Plots the numerically calculated collision points for a series of cluster 
    data trajectories which vary in their "acceleration" voltage 

    :param trajectories: cluster data trajectories
    :type trajectories: list
    :param voltages: voltages for each trajectory
    :type voltages: list
    :param wall_type: ID of the wall atoms
    :type wall_type: int
    :return: figure axis
    :rtype: mpl axis
    """
    collision_points = []
    for trajectory in trajectories:
        kes = cl_analysis.generate_droplet_kinetic_energy_timeseries(trajectory, wall_type)
        infls = cl_analysis.inflection_points(kes)
        collision_points.append(infls[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(voltages, collision_points, '--x')
    ax.set_xlabel('voltage in V')
    ax.set_ylabel('timestep of collision')
    plt.show()

    return ax

def plot_cluster_composition(trajectory, timestep, limit = 300):

    clusters = cl_analysis.filter_clusters_atom_composition(trajectory, timestep, limit)
    ax = sns.barplot(clusters, x="occurence", y="cluster", orient="y")
    ax.set_xlabel("occurence")
    plt.show()

    return ax 

def plot_cluster_composition_single_frame(trajectory, timestep, ax, limit = 300):

    clusters = cl_analysis.filter_clusters_atom_composition(trajectory, timestep, limit)
    sns.barplot(clusters, x="occurence", y="cluster", orient="y", ax=ax)
    ax.set_xlabel("occurence")
    plt.show()


def animate_cluster_composition(trajectory, animation_range, output, limit = 300):
    fig, ax = plt.subplots(1, 1)
    fig.set_tight_layout(True)

    x = range(animation_range[0], animation_range[1])

    def make_frame(i):
        ax.clear()
        plot_cluster_composition_single_frame(trajectory, i, ax, limit)

    # creating animation
    animation = FuncAnimation(fig,make_frame,frames=x)
    plt.close()

    from matplotlib.animation import PillowWriter
    # Save the animation as an animated GIF
    animation.save(output+".gif", dpi=600,
             writer=PillowWriter(fps=2))