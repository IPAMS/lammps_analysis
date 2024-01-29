import math 
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import LammpsAnalysis.cluster.cluster as cl 

def largest_value_column(dataframe, attribute):
    """
    Get largest value for a property

    :param dataframe: cluster data frame
    :type dataframe: dataframe
    :param attribute: property 
    :type attribute: string
    :return: max value for property
    :rtype: int
    """
    return dataframe[attribute].max()

def cluster_mass(dataframe, cluster_number):
    """
    Calculates mass of a given cluster 

    :param dataframe: cluster data frame
    :type dataframe: dataframe
    :param cluster_number: ID of cluster
    :type cluster_number: int
    :return: mass of cluster
    :rtype: float
    """
    filtered_df = dataframe.query('Cluster ==' + str(cluster_number))
    return filtered_df['Mass'].to_numpy().sum()

def calc_kinetic_energy(mass, velocity):
    """
    Calculates kinetic energy from mass and velocity (unit-preserving)

    :param mass: mass 
    :type mass: float
    :param velocity: velocity
    :type velocity: float
    :return: kinetic energy 
    :rtype: float
    """
    return 0.5 * mass * (velocity)**2

def mass_vel(mass, velocity):
    """
    Calculates mass-weighted velocity (SI units)

    :param mass: mass 
    :type mass: float
    :param velocity: velocity
    :type velocity: float
    :return: mass weighted velocity 
    :rtype: float
    """
    return mass * 1.66054e-27 * velocity * 1e5

def mass_pos(mass, position):
    """
    Calculates mass-weighted position (SI units)

    :param mass: mass 
    :type mass: float
    :param position: position
    :type position: float
    :return: mass weighted position 
    :rtype: float
    """
    return mass * 1.66054e-27 * position * 1e-10

def vec_abs(x, y, z):
    # TODO: switch with linalg.norm
    return math.sqrt(x**2 + y**2 + z**2)

def filter_species_no_wall(dataframe, wall_type):
    """
    Determine IDs of all species in a cluster trajectory that are not wall 

    :param dataframe: cluster trajectory
    :type dataframe: xarray
    :param wall_type: ID of wall atoms
    :type wall_type: int
    :return: unique species IDs
    :rtype: list
    """
    species = dataframe[0].loc[:,'Type'].to_pandas().pipe(lambda x: x[x != wall_type]).unique()
    return species

def filter_clusters_unique(trajectory, timestep):
    """
    Determine all clusters for a given frame in a cluster trajectory

    :param trajectory: cluster trajectory
    :type trajectory: xarray
    :param timestep: frame number
    :type timestep: int
    :return: unique cluster numbers
    :rtype: list
    """
    clusters =np.unique( trajectory[timestep].loc[:,'Cluster'])
    return clusters

def filter_clusters_unique_frame(dataframe):
    """
    Determine all clusters in a single cluster data frame

    :param dataframe: cluster data frame
    :type dataframe: dataframe
    :param timestep: frame number
    :type timestep: int
    :return: unique cluster numbers
    :rtype: list
    """
    clusters = dataframe.loc[:,'Cluster'].unique()
    return clusters

def count_atoms_clusters_unique(dataframe, timestep):
    selected_ts = dataframe[timestep].to_pandas()
    clusters = filter_clusters_unique_frame(selected_ts)
    atom_counts = []
    for cluster in clusters:
        atom_counts.append(selected_ts.query('Cluster ==' + str(cluster)).shape[0])

    ## get number of atoms in each cluster in a dataframe / how many clusters have x amount of atoms    
    return atom_counts

def count_atoms_clusters_unique_frame(dataframe):
    selected_ts = dataframe.to_pandas()
    clusters = filter_clusters_unique_frame(selected_ts)
    atom_counts = []
    for cluster in clusters:
        atom_counts.append(selected_ts.query('Cluster ==' + str(cluster)).shape[0])

    ## get number of atoms in each cluster in a dataframe / how many clusters have x amount of atoms    
    return atom_counts

def count_mass_clusters_unique(dataframe, timestep):
    selected_ts = dataframe[timestep].to_pandas()
    clusters = filter_clusters_unique_frame(selected_ts)
    mass_clusters = []
    for cluster in clusters:
        mass_clusters.append(cluster_mass(selected_ts, cluster))

    ## get number of atoms in each cluster in a dataframe / how many clusters have x amount of atoms    
    return mass_clusters

def count_mass_clusters_unique_frame(dataframe):
    selected_ts = dataframe.to_pandas()
    clusters = filter_clusters_unique_frame(selected_ts)
    mass_clusters = []
    for cluster in clusters:
        mass_clusters.append(cluster_mass(selected_ts, cluster))

    ## get number of atoms in each cluster in a dataframe / how many clusters have x amount of atoms    
    return mass_clusters

def cluster_count_trajectory(dataframe):
    ## get number of clusters per frame for whole dataset
    cluster_count = []
    for frame in dataframe:
        cluster_count.append(np.unique( frame.loc[:,'Cluster']).max())
    return cluster_count


def generate_droplet_kinetic_energy_timeseries(data, wall_type):
    ke_series = [droplet_kinetic_energy(frame.to_pandas(), wall_type) for frame in data]
    return ke_series

def inflection_points(kes):
    ## find inflection points 
    #  (should be one only, otherwise only first is used for collision detection)
    # smooth
    smooth = gaussian_filter1d(kes, 35, mode='nearest')
    # compute second derivative
    smooth_d2 = np.gradient(np.gradient(smooth))
    # find switching points
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    return infls

def get_all_cluster_distributions_in_space(dataframe, timestep):
    frame = cl.filter_frame(dataframe, timestep)
    clusters = filter_clusters_unique_frame(frame)
    coms = []
    masses = []
    for cluster in clusters:
        masses.append(cluster_mass(frame, cluster))
        coms.append(cluster_center_of_mass(frame, cluster))

    return np.asfarray(coms), np.array(masses), np.array(clusters, dtype=np.int8)



def cluster_kinetic_energy(dataframe, cluster_number, ev=True, total=False):
    filtered_df = dataframe.query('Cluster ==' + str(cluster_number))
    if(total):
        ke = filtered_df.apply(lambda x: calc_kinetic_energy(x.Mass, x.Velocity), axis=1)
        if(ev):
            return ke.to_numpy().sum()*6.242e18
        else:
            return ke.to_numpy().sum()
    else:
        total_mass = filtered_df['Mass'].to_numpy().sum() * 1.66054e-27
        vx = filtered_df['VX']
        vy = filtered_df['VY']
        vz = filtered_df['VZ']
        mass = filtered_df['Mass']
        weighted_vel_x = mass_vel(mass, vx)
        weighted_vel_y = mass_vel(mass, vy)
        weighted_vel_z = mass_vel(mass, vz)
        com_x = weighted_vel_x.sum() / total_mass
        com_y = weighted_vel_y.sum() / total_mass
        com_z = weighted_vel_z.sum() / total_mass

        ke = calc_kinetic_energy(total_mass, vec_abs(com_x, com_y, com_z))
    
        if(ev):
            return ke*6.242e18
        else:
            return ke

def droplet_kinetic_energy(dataframe, wall_type, ev=True, total=False):

    filtered_df = dataframe.query('Type !=' + str(wall_type))
    if(total):
        ke = filtered_df.apply(lambda x: calc_kinetic_energy(x.Mass, x.Velocity), axis=1)
        if(ev):
            return ke.to_numpy().sum()*6.242e18
        else:
            return ke.to_numpy().sum()
    else:
        total_mass = filtered_df['Mass'].to_numpy().sum() * 1.66054e-27
        vx = filtered_df['VX']
        vy = filtered_df['VY']
        vz = filtered_df['VZ']
        mass = filtered_df['Mass']
        weighted_vel_x = mass_vel(mass, vx)
        weighted_vel_y = mass_vel(mass, vy)
        weighted_vel_z = mass_vel(mass, vz)
        com_x = weighted_vel_x.sum() / total_mass
        com_y = weighted_vel_y.sum() / total_mass
        com_z = weighted_vel_z.sum() / total_mass

        ke = calc_kinetic_energy(total_mass, vec_abs(com_x, com_y, com_z))
    
        if(ev):
            return ke*6.242e18
        else:
            return ke

def cluster_center_of_mass(dataframe, cluster_number, si=False):
    filtered_df = dataframe.query('Cluster ==' + str(cluster_number))
    total_mass = filtered_df['Mass'].to_numpy().sum() * 1.66054e-27
    x = filtered_df['X']
    y = filtered_df['Y']
    z = filtered_df['Z']
    mass = filtered_df['Mass']
    weighted_x = mass_pos(mass, x)
    weighted_y = mass_pos(mass, y)
    weighted_z = mass_pos(mass, z)
    com_x = weighted_x.sum() / total_mass
    com_y = weighted_y.sum() / total_mass
    com_z = weighted_z.sum() / total_mass

    if(si):
        return [com_x, com_y, com_z]
    else:
        return [com_x*1e10, com_y*1e10, com_z*1e10]
    
def cluster_center_of_velocity_direction(dataframe, cluster_number, si=False):
    filtered_df = dataframe.query('Cluster ==' + str(cluster_number))
    total_mass = filtered_df['Mass'].to_numpy().sum() * 1.66054e-27
    vx = filtered_df['VX']
    vy = filtered_df['VY']
    vz = filtered_df['VZ']
    mass = filtered_df['Mass']
    weighted_vel_x = mass_vel(mass, vx)
    weighted_vel_y = mass_vel(mass, vy)
    weighted_vel_z = mass_vel(mass, vz)
    com_vx = weighted_vel_x.sum() / total_mass
    com_vy = weighted_vel_y.sum() / total_mass
    com_vz = weighted_vel_z.sum() / total_mass

    if(si):
        return [com_vx, com_vy, com_vz]
    else:
        return [com_vx*1e-5, com_vy*1e-5, com_vz*1e-5]