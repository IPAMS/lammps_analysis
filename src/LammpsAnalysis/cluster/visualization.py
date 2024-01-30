import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
import matplotlib.colors
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns
import numpy as np 
import LammpsAnalysis.cluster.analysis as cl_analysis
import pandas as pd

def plot_cluster_count_trajectory(dataframe):
    ## get number of clusters per frame for whole dataset
    cluster_count = cl_analysis.cluster_count_trajectory(dataframe)
    
    sc = plt.plot(cluster_count)
    plt.xlabel('timestep')
    plt.ylabel('cluster count')
    plt.show()

    return sc


def plot_droplet_kinetic_energy_timeseries(data, wall_type, collision_limit=5):
    kes = cl_analysis.generate_droplet_kinetic_energy_timeseries(data, wall_type)
    infls = cl_analysis.inflection_points(kes)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.plot(kes)
    plt.xlabel("timestep")
    plt.ylabel("kinetic energy of droplet in eV")

    ax2 = plt.axes([.56, .4, .3, .3])
    for i, infl in enumerate(infls, 1):
        ax.scatter(x=infl, y=kes[infl], color='green', label='Collision point', marker='x')
        ax2.scatter(x=infl, y=kes[infl], color='green', label='Collision point', marker='x')
    
    sc2 = ax2.plot(list(range(infls[0]-collision_limit,infls[0]+collision_limit)), 
                               kes[infls[0]-collision_limit:infls[0]+collision_limit])
    ax2.set_facecolor('white')
    ax.legend()
    mark_inset(ax, ax2, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.show()
    return sc, infls, kes

def plot_cluster_distribution(dataframe, timestep, savefig=False):    
    coms, masses, clusters = cl_analysis.get_all_cluster_distributions_in_space(dataframe, timestep)
    reldata = pd.DataFrame({'x':coms[:,0], 'y':coms[:,1], 'z':coms[:,2], 'mass':np.round(masses,3), 'cluster':np.round(clusters,0)})

    sc = sns.scatterplot(reldata, x='x', y='y', hue='cluster', size='mass', palette="magma", legend='full', size_norm=matplotlib.colors.LogNorm())

    box = sc.get_position()
    sc.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    sc.set_xlabel('x in Å')
    sc.set_ylabel('y in Å')
    sc.legend(loc='upper left', framealpha=1, bbox_to_anchor=(-0.12, -0.15), ncol=5)

    plt.show()
    if(savefig):
        plt.savefig('cluster_distribution.pdf', bbox_inches='tight')
    
    return sc

def plot_boxplot_cluster_sizes_trajectory(dataframe, limit = 40, step=10, offset=0):
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    result = pd.DataFrame()
    for i in range(0+offset, dataframe.shape[0], step):
        counts = cl_analysis.count_atoms_clusters_unique_frame(dataframe[i])
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
    return result, sc

def plot_boxplot_cluster_masses_trajectory(dataframe, limit = 200, step=10, offset=0):
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    result = pd.DataFrame()
    for i in range(0+offset, dataframe.shape[0], step):
        masses = cl_analysis.count_mass_clusters_unique_frame(dataframe[i])
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
    return result, sc

def plot_histogram_atoms_cluster(dataframe,timestep):
    counts = cl_analysis.count_atoms_clusters_unique(dataframe,timestep)

    reldata = pd.DataFrame({'atom count':counts})
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    sc = sns.histplot(reldata, x='atom count', log_scale=True, stat='count', ax=ax1)
    sc.set_ylabel('count #')

    ax2 = plt.axes([.46, .5, .4, .3])
    sc2 = sns.histplot(reldata[reldata < 40], x='atom count', stat='count', discrete=True, ax=ax2)
    ax2.set_facecolor('white')
    sc2.set_ylabel('count #')
    plt.show()
    return sc, sc2, fig1

def plot_histogram_mass_cluster(dataframe,timestep):
    masses = cl_analysis.count_mass_clusters_unique(dataframe,timestep)

    reldata = pd.DataFrame({'cluster masses':masses})
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    sc = sns.histplot(reldata, x='cluster masses', log_scale=True, stat='count', ax=ax1)
    sc.set_ylabel('count #')

    ax2 = plt.axes([.46, .5, .4, .3])
    sc2 = sns.histplot(reldata[reldata < 200], x='cluster masses', stat='count', discrete=True, ax=ax2)
    ax2.set_facecolor('white')
    sc2.set_ylabel('count #')
    plt.show()
    return sc, fig1


def plot_distribution_cluster_size_timeseries(dataframe, limit=40, step=10, offset=0):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    result = pd.DataFrame()
    for i in range(0+offset, dataframe.shape[0], step):
        counts = cl_analysis.count_atoms_clusters_unique_frame(dataframe[i])
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
    return result, sc


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
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

    angles = scattering_angles(trajectory, timestep, wall_type, limit, wall_vector)
    circular_hist(ax, angles)
    plt.show()

def plot_radial_distribution(trajectory, timestep, wall_type, limit, wall_vector=np.array([1, 0])):
    
    angles = scattering_angles(trajectory, timestep, wall_type, limit, wall_vector)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    circular_hist(ax, angles)
    plt.show()
    return fig, ax


def animate_radial_distribution(trajectory, wall_type, animation_range, output, limit, wall_vector=np.array([1, 0])):
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))

    x = range(animation_range[0], animation_range[1])

    def make_frame(i):
        ax.clear()
        radial_distribution_histogram(trajectory, i, wall_type, ax, limit, wall_vector)

    # creating animation
    animation = FuncAnimation(fig,make_frame,frames=len(x))
    plt.close()

    from matplotlib.animation import PillowWriter
    # Save the animation as an animated GIF
    animation.save(output+".gif", dpi=400,
             writer=PillowWriter(fps=5))
