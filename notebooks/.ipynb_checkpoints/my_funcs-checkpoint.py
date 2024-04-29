from IPython.display import display, Markdown
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
import matplotlib
from scipy import stats
import pandas as pd
from matplotlib.ticker import FuncFormatter

def print_with_font_size(text, fs=20):
    # Format the output with the desired font size
    print_output = f"<span style='font-size: {fs}px;'>{text}</span>"
    display(Markdown(print_output))

def rot_ticks(axs, rot, ha):
    for xlabels in axs.get_xticklabels():
        xlabels.set_rotation(rot)
        xlabels.set_ha(ha)

def fix_xticks(ax,ds):
    
    if  mdates.date2num(ds.time[-1]) - mdates.date2num(ds.time[0]) <= 2 :
        ax[0].xaxis.set_minor_locator(mdates.HourLocator(np.arange(0,24,3)))
        ax[0].xaxis.set_major_locator(mdates.HourLocator(np.arange(0,24,24)))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d %b"))
        ax[0].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax[0].set_xlim(ds.time[0] - np.timedelta64(1,'h'),ds.time[-1] + np.timedelta64(1,'h'))

    if  np.logical_and(mdates.date2num(ds.time[-1]) - mdates.date2num(ds.time[0]) > 2, mdates.date2num(ds.time[-1]) - mdates.date2num(ds.time[0]) <= 7):
        ax[0].xaxis.set_minor_locator(mdates.HourLocator([0,12]))
        ax[0].xaxis.set_major_locator(mdates.DayLocator(np.arange(1,32,1)))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
        ax[0].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax[0].set_xlim(ds.time[0] - np.timedelta64(1,'D'),ds.time[-1] + np.timedelta64(1,'D'))

    if  np.logical_and(mdates.date2num(ds.time[-1]) - mdates.date2num(ds.time[0]) > 7, mdates.date2num(ds.time[-1]) - mdates.date2num(ds.time[0]) <= 15):
        ax[0].xaxis.set_minor_locator(mdates.DayLocator(np.arange(1,32,1)))
        ax[0].xaxis.set_major_locator(mdates.DayLocator(np.arange(2,32,2)))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
        ax[0].xaxis.set_minor_formatter(mdates.DateFormatter(""))
        ax[-1].set_xlabel('2023')
        ax[0].set_xlim(ds.time[0] - np.timedelta64(1,'D'),ds.time[-1] + np.timedelta64(1,'D'))

    if  np.logical_and(mdates.date2num(ds.time[-1]) - mdates.date2num(ds.time[0]) > 15, mdates.date2num(ds.time[-1]) - mdates.date2num(ds.time[0]) <= 30):
        ax[0].xaxis.set_minor_locator(mdates.DayLocator(np.arange(1,32,1)))
        ax[0].xaxis.set_major_locator(mdates.DayLocator([5,10,15,20,25,30]))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d\n%b"))
        #ax[0].xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
        ax[0].set_xlim(ds.time[0] - np.timedelta64(2,'D'),ds.time[-1] + np.timedelta64(2,'D'))

    if  mdates.date2num(ds.time[-1]) - mdates.date2num(ds.time[0]) > 30:
        ax[0].xaxis.set_major_locator(mdates.MonthLocator())
        ax[0].xaxis.set_minor_locator(mdates.DayLocator([1,5,10,15,20,25]))
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d\n%B"))
        ax[0].xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
        ax[0].set_xlim(ds.time[0] - np.timedelta64(3,'D'),ds.time[-1] + np.timedelta64(3,'D'))

    rot_ticks(ax[-1],0,'center')
    ax[0].set_xlabel('')

def calc_bias(ds):
    return np.nanmean(ds)

def calc_r2(ds, var1, var2):
    _,_,r,_,_ = stats.linregress(ds[var1],ds[var2])
    return r**2

def calc_rmse(ds):
    return np.sqrt(np.mean(np.square(ds)))
    
def circ_cmap(cmap_name, num_colors=4):
    """
    Create a circular colormap with a specified number of colors.

    Parameters:
        cmap_name (str): Name of the original colormap to use for interpolation.
        num_colors (int): Number of colors in the circular colormap.

    Returns:
        numpy.ndarray: Circular colormap as an array of RGB values.
    """
    # Get the original colormap
    original_cmap = plt.get_cmap(cmap_name)

    # Interpolate colors from the original colormap
    interpolated_colors = original_cmap(np.linspace(0, 1, num_colors, endpoint=False))

    # Extract RGB values from the interpolated colors
    rgb_values = np.array([x[:3] for x in interpolated_colors])

    # Repeat each color once
    repeated_colors = np.repeat(rgb_values, 2, axis=0)

    # Roll one step to the left to create a circular effect
    circular_colors = np.roll(repeated_colors, shift=-1, axis=0)

    custom_cmap = ListedColormap(circular_colors)
    
    return custom_cmap

def calc_stats(ds, pimpim=True):
    """
    Calculate statistics for specified wind bins and variables.

    Parameters:
    - ds (xarray.Dataset): Dataset containing the data.
    - pimpim (bool): Flag to determine the platform and variables.

    Returns:
    - pd.DataFrame: DataFrame containing the calculated statistics.
    """
   
    # Set variables and platform based on the 'pimpim' flag
    if pimpim == True:
        variables = ['u10', 'wind_dir', 't2m', 'qref', 'qair', 'qsea', 'rh']
        wind_bins = [0, 0, 5, 0, 0, 0, 5]  # Modify this according to your specific wind bins
        platform = 'sbpp'
    else:
        variables = ['u10', 'wind_dir', 't2m']
        wind_bins = [0, 0, 5]  # Modify this according to your specific wind bins
        platform = 'sbk'
    
    # Create an empty list to store statistics DataFrames
    ds_stats = []
    
    # Loop through wind bins and variables
    for wind_bin, var in zip(wind_bins, variables):
        # Statistics mask
        msk = np.intersect1d(
            np.where(np.isfinite(ds.where(ds['u10_ship'] > wind_bin)['u10_ship'])),
            np.where(np.isfinite(ds.where(ds['u10_ship'] > wind_bin)[f'u10_{platform}']))
        )
    
        # Statistics calculations
        s, i, r, p, _ = stats.linregress(ds[f'{var}_{platform}'][msk], ds[f'{var}_ship'][msk])
        rmse = calc_rmse(-ds[f'{var}_ship'][msk] + ds[f'{var}_{platform}'][msk]).values.round(2)
        bias = calc_bias(-ds[f'{var}_ship'][msk] + ds[f'{var}_{platform}'][msk]).round(2)
        rmse_bias = calc_rmse(-ds[f'{var}_ship'][msk] + ds[f'{var}_{platform}'][msk] - bias).values.round(2)
    
        # Create a data dictionary
        data = {
            'stats': ['slope', 'intercept', 'r2', 'p_value', 'rmse', 'bias', 'rmse_bias'],
            var: [s, i, r**2, p, rmse, bias, rmse_bias]
        }
    
        # Create a DataFrame for the current variable
        var_df = pd.DataFrame.from_dict(data, orient='index').T.set_index('stats')
    
        # Append the DataFrame to the list of statistics
        ds_stats.append(var_df)
    
    # Merge the list of DataFrames
    ds_stats = pd.concat(ds_stats, axis=1)
    return ds_stats


def update_params(fontsize=20):
    """
    Update font and tick parameters for matplotlib.

    This function updates the font and tick parameters for matplotlib to achieve a consistent style.
    It also adds custom fonts to the font manager.

    Usage:
        Call this function to update the matplotlib parameters in your notebook.

    Example:
        update_params()

    """

    font_dirs = ["/Users/xedhjo/Library/Fonts/"]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    
    # Add the custom font to the font manager
    font_manager.fontManager.addfont("/Users/xedhjo/Library/Fonts/Alice-Regular.ttf")

    # Specify the custom font family
    font = {'family': 'Alice',
            'size': fontsize}

    tick_params = {
        'xtick.major.size': 4,
        'xtick.major.width': 1,
        'xtick.minor.size': 4,
        'xtick.minor.width': 1,
        'ytick.major.size': 4,
        'ytick.major.width': 1,
        'ytick.minor.size': 4,
        'ytick.minor.width': 1
    }

    matplotlib.rc('font', **font)
    matplotlib.rcParams.update(tick_params)

    
    

# Function to format ticks on the plot axes
def format_ticks(ax, data, axis='x'):
    """
    Formats ticks on the plot axes for distance or percentage values.

    Parameters:
    - ax: The axis object for the plot.
    - data: Specifies the type of data to format ("m" for meters, "km" for kilometers, "percent" for percentages, "fraction" for fractions).
    - axis: Specifies whether to format ticks on the 'x' or 'y' axis. Default is 'x'.

    Returns:
    None
    """
    # Define custom formatter functions
    # Function to format distance values in kilometers or meters
    def distance_formatter(x, pos, as_km=True):
        if as_km:
            return f'{x:.0f} km'
        else:
            return f'{x / 1000:.0f} km'

    # Function to format percentage values in the specified range
    def percent_formatter(x, pos, as_percent=True):
        if as_percent:
            return f'{x:.0f}%'
        else:
            return f'{x*100:.0f}%'

    # Determine which formatter function to use
    if data == "m":
        formatter = distance_formatter
        as_km = False
    elif data == "km":
        formatter = distance_formatter
        as_km = True
    elif data == "percent":
        formatter = percent_formatter
        as_percent = True
    elif data == "fraction":
        formatter = percent_formatter
        as_percent = False
    else:
        raise ValueError("Invalid value for 'data'. Use 'm', 'km', 'percent', or 'fraction'.")

    # Apply formatter based on the specified axis
    if axis == 'x':
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: formatter(x, pos, as_km if data in ["m", "km"] else as_percent)))
    elif axis == 'y':
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: formatter(x, pos, as_km if data in ["m", "km"] else as_percent)))
    else:
        raise ValueError("Invalid value for 'axis'. Use 'x' or 'y'.")







