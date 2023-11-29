import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.patches as mpatches
import os
from scipy.stats import gmean, f_oneway

from process import *

#Exclude possible entries in data that do not represent tasks
excluded_tasks = ('Total', 't_total', 't_algo', 't_frame', 
                   'Image Resize',
                  'ATE', 'n_frames')

colors = ["red", "green", "blue", "magenta", "yellow", "sandybrown", "greenyellow", "violet", "darkolivegreen"]

class GroupedFigure:
    """
    Class for managing group of pyplot figures and setting the same x/y ranges for all of them.
    """
    def __init__(self, group_x, group_y) -> None:
        self.group_x = group_x
        self.group_y = group_y
        self.axes = []
        self.figures = []
        self.savedirs = []
        self.ylims = []
        self.xlims = []
    
    
    def new_plot(self, function, *args, **kwargs):
        """
        Adds a new figure using the provided function and arguments to pass to it.
        """
        savedir = function(*args, **kwargs, save=False) #Always set save to false so it don't get saved twice by mistake (the latter save from the class would overwrite anyway)
        self.figures.append(plt.gcf())
        self.savedirs.append(savedir)
        ax = plt.gca()
        self.axes.append(ax)
        self.ylims.append(ax.get_ylim())
        self.xlims.append(ax.get_xlim())
    
    
    def regroup_axes(self):
        """
        Call when all figures have been added to compute common min/max and apply it.
        """
        ylims = np.asarray(self.ylims)
        xlims = np.asarray(self.xlims)
        
        if self.group_x:
            min_x = np.min(xlims[:,0])
            max_x = np.max(xlims[:,1])
            
            for ax in self.axes:
                ax.set_xlim((min_x, max_x))

        if self.group_y:
            min_y = np.min(ylims[:,0])
            max_y = np.max(ylims[:,1])
            
            for ax in self.axes:
                ax.set_ylim((min_y, max_y))
    
    
    def save_figures(self):
        """
        Saves the figure to a file (directory given by the plotting functions)
        """
        
        for savedir, fig in zip(self.savedirs, self.figures):
            fig.savefig(savedir)
            plt.close(fig)


def plot_timeline_tasks(data, dataset, run):

    fig, ax = plt.subplots(figsize=(12, 5))
    
    data = data[dataset]['runs'][run]
    for task, task_times in reversed(data.items()):
        if task in excluded_tasks:
            continue
        ax.plot(task_times, label=task)
    
    plt.legend()
                

def plot_histogram_frame_time(data, labels, normalize=False, savedir=None, save=False):
    """ Plot the histogram of the time spent per frame for diferent data passed as list"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if not isinstance(data, list):
        data = [data]
        labels = [labels]
    
    #NOTE: Does it make sense to use the mean instad to compute the histogram? I will use ALL the measures from all the runs so I don't average the peaks
    
    for times, label in zip(data, labels):
        counts, bins = np.histogram(x)
        ax.stairs(counts, bins, label=label)
        #ax.hist(times, bins='auto', histtype='step', label=label, density=normalize)
            
    
    fig.legend()

    title_str = "Histogram"
    ax.set_title(title_str)
    ax.set_xlabel("Time elapsed per frame (ms)")
    ax.set_ylabel(f"%s Frequency" % ("Normalized" if normalize else ""))
    
    if savedir is not None:
        #TODO: THIS IS FROM THE OLD HISTOGRAM FUNCTION. FIX.
        file_str = "Histogram_%s" #% (get_reformated_simple_name(data))
        
        file_str += ".png"
        file_str = os.path.join(savedir, file_str)
        
        if save:
            fig.savefig(file_str)
            plt.close()
        
        return file_str
    

def plot_cdf_frame_time(data, labels, normalize=False, savedir=None, save=False):
    """ Plot the Cumulative Distribution Function of the time spent per frame for diferent data pass as list"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if not isinstance(data, list):
        data = [data]
        labels = [labels]
        
    for times, label in zip(data, labels):
        n = len(times)
        x = np.sort(times)
        y = np.arange(1, n+1) / n

        ax.plot(x, y, label=label)
            
    
    fig.legend()

    title_str = "Cumulative Distribution"
    ax.set_title(title_str)
    ax.set_xlabel("Time elapsed per frame (ms)")
    ax.set_ylabel(f"Cumulative frequency")
    
    if savedir is not None:
        #TODO: THIS IS FROM THE OLD HISTOGRAM FUNCTION. FIX.
        file_str = "Histogram_%s" #% (get_reformated_simple_name(data))
        
        file_str += ".png"
        file_str = os.path.join(savedir, file_str)
        
        if save:
            fig.savefig(file_str)
            plt.close()
        
        return file_str


def plot_tasks_bars(data, mode, version=None):
    """Plot the time taken by each task on stacked horizontal bars.
    Mode options are:
        'runs': Compare individual runs of the same dataset on different figures
        'datasets': Compare average of runs for each dataset. Optionally provide a version name for titles."""
    
    def draw_bars():
        cum = np.zeros_like(tasks_time[0]) #size of the number of runs
        legend_array = list()
        for task_time, task_std in zip(tasks_time, tasks_std):
            if task in excluded_tasks:
                continue
            legend_array.append(ax.barh(instances_names, task_time, left=cum, xerr=task_std, capsize=2))
            cum += task_time
        
        fig.legend(legend_array, tasks)
    
    match mode:
        case 'runs':
            for dataset in data:
                fig, ax = plt.subplots(figsize=(12, 5))
                
                tasks = [taskname for taskname in dataset['runs'][0].keys() if taskname not in excluded_tasks]
                tasks_time = [list() for _ in tasks]
                tasks_std = [list() for _ in tasks]
                
                instances_names = list()
    
                for i, run in enumerate(dataset['runs']):
                    instances_names.append(f'Run {i}')
                    for t, task in enumerate(tasks):
                        tasks_time[t].append(np.nanmean(run[task]))
                        tasks_std[t].append(np.nanstd(run[task]))
                
                draw_bars()
                
                ax.set_title(f'Run comparison for {dataset["name"]} dataset{f" for version {version}" if version != None else ""}')
                ax.set_xlabel("Time elapsed per frame (ms)")
        case 'datasets':
            fig, ax = plt.subplots(figsize=(12, 5))
            
            tasks = [taskname for taskname in data[0]['runs'][0].keys() if taskname not in excluded_tasks]
            tasks_time = [list() for _ in tasks]
            tasks_std = [list() for _ in tasks]
            
            instances_names = list()
            
            for dataset in data:
                instances_names.append(dataset['name'])
                for t, task in enumerate(tasks):
                    times_mean = list()
                    times_std = list()
                    for run in dataset['runs']:
                        times_mean.append(np.nanmean(run[task]))
                        times_std.append(np.nanstd(run[task]))
                        
                    times_std = np.array(times_std)
                    tasks_time[t].append(np.mean(times_mean))
                    tasks_std[t].append(np.sqrt(np.sum(times_std**2))/len(times_std)) #error propagation
            
            draw_bars()
            
            ax.set_title(f'Dataset comparison{f" for version {version}" if version != None else ""}')
            ax.set_xlabel("Time elapsed per frame (ms)")
                
            
            
        case _:
            print("Invalid plotting mode")
            return
        

def plot_speedup(datas, titles, mode):
    """Plot speedup information about the datas
    mode can be set to:
        'absolute': Print absolute time taken
        'frame': Print time per frame
        'speedup': Speedup of all datas agaist the first 1
        'error': Error of all datas
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    match mode:
        case 'absolute':
            xlabel_text = "Execution time (s)"
            title_text = "Total execution time version comparison"
        case 'frame':
            xlabel_text = "Time elapsed per frame (ms)"
            title_text = "Mean execution time per frame version comparison"
        case 'frequency':
            xlabel_text = "Frames per second (Hz)"
            title_text = "Frequency version comparison"
        case 'speedup':
            xlabel_text = "Speedup"
            title_text = f"Speedup comparison Vs {titles[0]} version"
        case 'error':
            xlabel_text = "ATE"
            title_text = f"Absolute Trajectory Error (ATE) comparison"
        case _:
            print("Invalid plotting mode")
            return
        
    #Datasets consistency checkout
    dataset_names = [dataset['name'] for dataset in datas[0]]
    for data in datas:
        for i, dataset in enumerate(data):
            assert dataset_names[i] == dataset['name']
    
    if mode in ('absolute', 'frame', 'frequency'):
        times = list()
        for data in datas:
            times_data = list()
            for dataset in data:
                times_data.append(list())
                for run in dataset['runs']:
                    #Convert to time per frame or from seconds from milli depending on frame/absolute
                    val = run['t_algo'] / (run["n_frames"] if mode=='frame' else 1000)
                    if mode=='frequency':
                        val = run["n_frames"]/val
                    times_data[-1].append(val) 
            
            times.append(times_data)
                
        num_datasets = len(dataset_names)
        num_experiments = len(datas)
        width = 0.6
        gap = 3
        
        for i in range(num_experiments):
            positions = [j*(num_experiments + gap) + i for j in range(num_datasets)]
            ax.boxplot(times[i], positions=positions, widths=width, patch_artist=True, boxprops=dict(facecolor=colors[i]), vert=False)
        
        legend_array = [mpatches.Patch(color=colors[i], label=titles[i]) for i in range(num_experiments)]
        fig.legend(handles=legend_array)
        
    elif mode == 'speedup':
        reference = datas[0]
        
        #More elegant way to make the same lists as above
        reference_means = np.array([np.mean([run['t_algo'] / 1000 for run in dataset['runs']]) for dataset in reference]) 
        reference_stds = np.array([np.std([run['t_algo'] / 1000 for run in dataset['runs']]) for dataset in reference])
        
        for i, data in enumerate(datas[1:]):            
            data_means = np.array([np.mean([run['t_algo'] / 1000 for run in dataset['runs']]) for dataset in data])
            data_stds = np.array([np.std([run['t_algo'] / 1000 for run in dataset['runs']]) for dataset in data])
            
            speedups = reference_means / data_means
            speedups_std = np.sqrt((data_stds/data_means)**2 + (reference_stds/reference_means)**2)
            
            num_datasets = len(dataset_names)
            num_experiments = len(datas) - 1
            gap = 4
            
            positions = [j*(num_experiments + gap) + i for j in range(num_datasets)]
            ax.errorbar(speedups, positions, xerr=speedups_std, label=titles[i+1], fmt='o', capsize=2)
            
            fig.legend()
    elif mode == 'error':
        error = list()
        for data in datas:
            error_data = list()
            for dataset in data:
                error_data.append(list())
                for run in dataset['runs']:
                    #Convert to time per frame or from seconds from milli depending on frame/absolute
                    val = run['ATE']
                    error_data[-1].append(val) 
            
            error.append(error_data)
        
        #Compute p-value

        for i in range(len(error[0])):
            ed = [e[i] for e in error]
            s, p = f_oneway(*ed)
            dataset_names[i] += f" (p={round(p, 2)})"

        num_datasets = len(dataset_names)
        num_experiments = len(datas)
        width = 0.6
        gap = 3
        
        for i in range(num_experiments):
            positions = [j*(num_experiments + gap) + i for j in range(num_datasets)]
            ax.boxplot(error[i], positions=positions, widths=width, patch_artist=True, boxprops=dict(facecolor=colors[i]), vert=False)
        
        legend_array = [mpatches.Patch(color=colors[i], label=titles[i]) for i in range(num_experiments)]
        fig.legend(handles=legend_array)
    
    ax.set_yticks([(num_experiments+gap)*i + (num_experiments-1)/2 for i in range(num_datasets)])
    ax.set_yticklabels(dataset_names)
        
    ax.set_ylabel("Datasets")
    ax.set_xlabel(xlabel_text)
    ax.set_title(title_text)
    ax.invert_yaxis()


def draw_boxes(data, time_unit_string, stage_names = None, title = None, start_item=0, number_y_labels=35):
    """Pipeline drawer"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    n_stages = len(data[0])
    for row, item in enumerate(data):
        if row < start_item:
            continue
        ax.broken_barh(item, (row - 0.25, 0.5), facecolors=colors[:n_stages])
    
    #Bolzano theorem (kinda...) The best padding is one of the two that are closest to 0
    e = np.inf
    for p in range(1, len(data)-start_item):
        n_yticks = len(range(start_item,len(data), p))
        last_e = e
        e = n_yticks - number_y_labels
        if e < 0:
            break
    padding = p if -e < last_e else p - 1
    
    ax.set_ylim(len(data), start_item-0.5)
    ax.set_xlim(0, max([subitem[0] + subitem[1] for subitem in item for item in data])) #TODO: Dynamically adjust min depending on start_item 
    ax.set_xlabel('Execution Timelapse (%s)' % (time_unit_string))
    ax.set_yticks([i for i in range(start_item,len(data), padding)])
    ax.set_yticklabels([f'Item {i+1}' for i in range(start_item, len(data), padding)])

    
    legend_array = [mpatches.Patch(color=colors[i], label=f"Stage {i}" if stage_names is None else stage_names[i]) for i in range(n_stages)]
    plt.legend(handles=legend_array)
    
    if title is not None:
        ax.set_title(title)


def compare_tasks(datas, titles):
    """Compares the task cost for the diferent versions provided.
    One figure per dataset."""
    
    #Datasets consistency checkout (less restrictive than before. Allows for not using datasets if not included on the first data)
    datasets = [dataset['name'] for dataset in datas[0]]
    for data in datas:
        datasets_data = [d["name"] for d in data]
        for dataset in datasets:
            assert dataset in datasets_data
    
    tasks = [taskname for taskname in datas[0][0]['runs'][0].keys() if taskname not in excluded_tasks]
    
    for dataset_idx in range(len(datasets)):
        fig, ax = plt.subplots(figsize=(12, 5))
        times = list()
        stds = list()
        for data in datas:
            dataset_tasks = list()
            for run in data[dataset_idx]['runs']:
                dataset_tasks.append(list())
                for task in tasks:
                    dataset_tasks[-1].append(run[task])

            #Operations first on dim2 (all frames) and then on dim 0 (all runs) and you end up with a vector with size n_tasks
            times.append(np.nanmean(np.nanmean(dataset_tasks, axis=2), axis=0))
            stds.append(np.sqrt(np.nansum(np.nanstd(dataset_tasks, axis=2)**2, axis=0))/len(dataset_tasks))
            
        num_tasks = len(tasks)
        num_experiments = len(datas)
        width = 0.6
        gap = 1
        
        for i in range(num_experiments):
            positions = [j*(num_experiments + gap) + i for j in range(num_tasks)]
            ax.barh(positions, times[i], xerr=stds[i], color=colors[i], capsize=2)        
        ax.set_yticks([(num_experiments+gap)*i + (num_experiments-1)/2 for i in range(num_tasks)])
        ax.set_yticklabels(tasks)
    
        ax.set_ylabel("Tasks")
        ax.set_xlabel("Time elapsed per frame (ms)")
        ax.set_title(f"Task breakdown by version comparison for dataset {datasets[dataset_idx]}")
        ax.invert_yaxis()
        
        legend_array = [mpatches.Patch(color=colors[i], label=titles[i]) for i in range(num_experiments)]
        plt.legend(handles=legend_array)


def compare_tasks_sequential_pipeline(data_sequential_full, data_pipeline_full, equivalences, tasks,):
    """Compares the task latency between two entries: one sequential and one pipeline.
    The equivalences variable is the tasks that add to each stage (can be more than one).
    
    One figure per dataset"""
    
    def compute_mean_and_std(data):
        "From an array of runs. Returns mean and std with error propagation per run"
        stds_runs = np.std(data, axis=1)
        return np.nanmean(data), np.sqrt(np.nansum(stds_runs**2)/len(stds_runs))
    
    
    for data_sequential, data_pipeline in zip(data_sequential_full, data_pipeline_full):
        fig, ax = plt.subplots(figsize=(12, 5))
        
        assert data_sequential['name'] == data_pipeline['name']
        
        means_sequential = list()
        stds_sequential = list()
        means_pipeline = list()
        stds_pipeline = list()
        
        for stage, tasks_sequential in enumerate(equivalences):
            time_sequential = []
            for run in data_sequential['runs']:
                task_time = np.zeros_like(run[tasks_sequential[0]]) 
                for task_sequential in tasks_sequential: #Add all the equivalences time
                    task_time += run[task_sequential]
                time_sequential.append(task_time)
            
            time_pipeline = []
            for run in data_pipeline['runs']:
                time_pipeline.append(np.array([item[stage][1] for item in run]))

            mean_sequential, std_sequential = compute_mean_and_std(time_sequential)
            means_sequential.append(mean_sequential)
            stds_sequential.append(std_sequential)
            mean_pipeline, std_pipeline = compute_mean_and_std(time_pipeline)
            means_pipeline.append(mean_pipeline)
            stds_pipeline.append(std_pipeline)

        num_experiments = 2
        num_tasks = len(tasks)
        width = 0.6
        gap = 1
            
        for i, data in enumerate(((means_sequential, stds_sequential), (means_pipeline, stds_pipeline))):
            positions = [j*(num_experiments + gap) + i for j in range(num_tasks)]
            ax.barh(positions, data[0], xerr=data[1], color=colors[i], capsize=2)
        
        ax.set_yticks([(num_experiments+gap)*i + (num_experiments-1)/2 for i in range(num_tasks)])
        ax.set_yticklabels(tasks)
        
        legend_array = [mpatches.Patch(color=colors[1], label="Pipeline"),
                        mpatches.Patch(color=colors[0], label="Sequential")]
        plt.legend(handles=legend_array)
        
        ax.set_title(f"Tasks comparison for dataset {data_sequential['name']}")



def plot_number_tokens_pipeline(datas, labels, reference=None, task = "Total", mode=None):
    """ If reference is given, it computes speedup.
        If mode is 'datasets' all datasets are plotted, otherwise only the geometric mean
        If only one data is given in datas, the boxplot of all datasets is also plotted
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    if task == "Total":
        reference_task = "t_total"
        ax.set_title("Total execution time")
    elif task == "Algo":
        reference_task = "t_algo"
        ax.set_title("Algorithm execution time")
    
    if reference is not None:
        reference_means = np.array([np.mean([run[reference_task] / 1000 for run in dataset['runs']]) for dataset in reference]) 
        reference_stds = np.array([np.std([run[reference_task] / 1000 for run in dataset['runs']]) for dataset in reference])
    
        for data, dataname in zip(datas, labels):
            speedups = list()
            speedups_std = list()
            for i, dataset in enumerate(data):
                n_tokens = sorted(data[dataset][task].keys())
                means = np.array([np.mean(data[dataset][task][n_token]) for n_token in n_tokens])
                stds = np.array([np.std(data[dataset][task][n_token]) for n_token in n_tokens])
                #
                
                speedups.append(reference_means[i] / means)
                speedups_std.append(np.sqrt((stds/means)**2 + (reference_stds[i]/reference_means[i])**2))
            
                #ax.errorbar(n_tokens, speedups[-1], speedups_std[-1], capsize=2, label=dataset+dataname)
            speedups = np.array(speedups)
            speedups_std = np.array(speedups_std)
            
            gmeans = gmean(speedups)
            gstds = gmeans * np.sqrt(np.sum((speedups_std/speedups)**2, axis=0)) / speedups_std.shape[0]
            
            ax.errorbar(n_tokens, gmeans, gstds, label="Geometric mean"+dataname, capsize=2, lw=3 if mode=='datasets' else 2)
            
            if (len(datas) == 1): #Only show boxplot if only one data is given
                ax.boxplot(speedups, positions=n_tokens)
            
                if mode=='datasets':
                    for i, dataset in enumerate(data):
                        ax.errorbar(n_tokens, speedups[i], speedups_std[i], lw=1, capsize=2,label=dataset)            
    else:
        raise(NotImplementedError) #Total execution time
    
    ax.set_ylabel("Speedup")
    ax.set_xlabel('Number of pipeline tokens')
    
    plt.legend(ncol=2)
    
def plot_histogram_tasks(data, normalize=False):
    """Plot the histogram of time taken by each task.
        One dataset per figure"""
    #TODO??: One task per figure with all datasets. See differences per task in dataset?
    
    
    for dataset in data:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        tasks = [taskname for taskname in dataset['runs'][0].keys() if taskname not in excluded_tasks]

        for t, task in enumerate(tasks):
            task_times = list()
            for i, run in enumerate(dataset['runs']):
                task_times.extend(run[task])
            
            ax.hist(task_times, bins='auto', histtype='step', label=task, density=normalize)

        fig.legend()

        title_str = f"Histogram for {dataset['name']}"
        ax.set_title(title_str)
        ax.set_xlabel("Time elapsed per frame (ms)")
        ax.set_ylabel(f"%s Frequency" % ("Normalized" if normalize else ""))



if __name__ == "__main__":
    with open(file_dir, "r") as f:
        lines = f.readlines()
    
    t_frame, t_load_image, t_track, t_wait, t_algo, T = process_orbslam_output_file(lines)
    
    plot_histogram_frame_time(t_track, "Frame")
    
    plt.show()
    
    