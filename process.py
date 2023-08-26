import matplotlib.pyplot as plt
import numpy as np
import re
import os

from plotutils import *

#file_dir=r'run_data/TrackingTimeStats.txt'
file_dir = r'Results/test_comeback/output.dat'


def process_orbslam_dir(directory, analysis='timestamp_legacy'):
    """Process a directory with several directories of structure dataset/run_x
    The analysis variable can be set to:
        'timestamp': Analyze using manual timestamps on the code.
        'orbslam': Analyze using time data of the ORBSLAM timers.
        'timestamp_legacy': Same as timestamp but to reproduce older examples.
        "pipeline": Gives all the pipeline data
    """
    
    match analysis:
        case "timestamp":
            analysis_file = "orbslam3_output.log"
        case "orbslam":
            analysis_file = "TrackingTimeStats.txt"
        case "timestamp_legacy":
            analysis = 'timestamp' #Change it for after (function is the same, just change the name of the file)
            analysis_file = "orbslam3_new.log"
        case "pipeline":
            analysis_file = "test.dat"
        case _:
            print("Invalid analysis mode")
            return None

    data = list()
    for dataset in os.listdir(directory):
        data_dict = {"name":dataset, "runs":list()}
        
        for run in os.listdir(os.path.join(directory, dataset)):
            file_path = os.path.join(directory, dataset, run, analysis_file)
            print("\t", file_path)
            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    if analysis == 'timestamp':
                        t_frame, t_load_image, t_track, t_wait, t_algo, t_total = process_orbslam_output_file(lines)

                        sub_dict = dict()
                        sub_dict["t_frame"] = t_frame
                        sub_dict["t_load_image"] = t_load_image
                        sub_dict["t_track"] = t_track
                        sub_dict["t_wait"] = t_wait
                        sub_dict["t_algo"] = t_algo
                        sub_dict["t_total"] = t_total

                    elif analysis == 'orbslam':
                        sub_dict = process_tracking_time_stats(lines)
                        
                        T = sub_dict['Total']
                        t = np.zeros_like(T)
                        for k in sub_dict.keys():
                            if k != 'Total':
                                t += sub_dict[k]

                        sub_dict["Other"] = T-t
                    elif analysis == 'pipeline':
                        sub_dict = process_pipeline_times_output_file(lines, 1e3)
                    
                    file_path = os.path.join(directory, dataset, run, "SessionInfo.txt")
                    print("\t", file_path)
                    with open(file_path, "r") as f_info:
                        lines_info = f_info.readlines()

                        sub_dict["ATE"] = float(lines_info[3].split(':')[1].split(',')[0])
                        sub_dict["n_frames"] = int(lines_info[4].split(':')[1])
                    

                    data_dict["runs"].append(sub_dict)
            except(FileNotFoundError):
                print("\tNOT FOUND")        
            
        data.append(data_dict)
    
    return data


def process_orbslam_output_file(lines):
    times = []
    times_load_image = []
    times_track = []
    times_wait = []
    i = 0
    started = False
    t_algo = None
    t_exec = None
    last_end_track = 0
    
    while i<len(lines):
        #TODO: Convert to switch/case
        fields = lines[i].split('\t')
        if fields[0] == 'START':
            t0 = int(fields[1])
        if fields[0] == 'ALGO_START':
            t1 = int(fields[1])
            last_frame = t1
            started = True
        
        if started:
            if fields[0] == 'FRAME':
                t = int(fields[1])
                times.append(t - last_frame)
                last_frame = t
                times_wait.append(t-last_end_track)
            
            elif fields[0] == 'LOAD_IMAGE':
                t = int(fields[1])
                last_load_image = t
                times_load_image.append(t-last_frame)
                
            elif fields[0] == 'END_TRACK':
                t = int(fields[1])
                last_end_track = t
                times_track.append(t-last_load_image)
                
            elif fields[0] == 'ALGO_END':
                t = int(fields[1])
                t_algo = t - t1
                
            elif fields[0] == 'END':
                t = int(fields[1])
                t_exec = t - t0
        
        i += 1
    
    times.append(t_algo + t1 - last_frame)
    times.pop(0)
    times_wait.append(t_algo + t1 - last_end_track)
    times_wait.pop(0)
    
    times = np.array(times, dtype=float)
    times_load_image = np.array(times_load_image, dtype=float)
    times_track = np.array(times_track, dtype=float)
    times_wait = np.array(times_wait, dtype=float)
    
    #Convert from ns to ms
    times /= 1e6
    times_load_image /= 1e6
    times_track /= 1e6
    times_wait /= 1e6
    t_algo /= 1e6
    t_exec /= 1e6
    
    return times, times_load_image, times_track, times_wait, t_algo, t_exec


def process_tracking_time_stats(lines):
    data_dict = {}
    
    #Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]
    fieldsh= lines[0][1:].split(',')
    fieldsh = [field.strip().split('[')[0] for field in fieldsh] #Remove leading space and unit measure
    data_dict = {field:[] for field in fieldsh}
    
    for line in lines[1:]:
        fields = line.split(',')
        for fieldh, field in zip(fieldsh, fields):
            val = float(field.strip())
            if val > 10000 or val < 0: #Ignore broken cases (sometimes the value goes crazy??? A limit of 10s should strongly cover all real cases)
                val = np.nan
            data_dict[fieldh].append(val)
    
    for fieldh in data_dict.keys():
        data_dict[fieldh] = np.nan_to_num(np.array(data_dict[fieldh]), nan=np.nan) #Convert nans to np.nan to correctly ignore them in means
    
    return data_dict
            


def process_pipeline_times_output_file(lines, time_factor_reduction=1, max_items=np.inf):
    lines = lines[1:] #Ignore header
    
    #Search for number of items/stages
    items = [int(line.split('\t')[0]) for line in lines]
    stages = [int(line.split('\t')[1]) for line in lines]
    n_items = max(items) + 1
    n_stages = max(stages) + 1
    
    n_items = min(n_items, max_items)
    
    data = [[-1]*n_stages for _ in range(n_items)]
    
    for line in lines:
        item, stage, t_start, t_end = [int(_) for _ in line.split('\t')]
        
        if item >= n_items:
            continue #Skip if over max_items
        
        t_start /= time_factor_reduction
        t_end /= time_factor_reduction
        
        data[item][stage] = (t_start, t_end-t_start)
    
    return data

def process_and_plot_pipeline_tokens(dir):
    #with open("pipeline_tests/n_tokens/output.dat", "r") as f: Example file name
    with open(dir, "r") as f:
        lines = f.readlines()
        
        n = []
        t = []
        for line in lines:
            fields = line.split(' ')
            n.append(int(fields[0].strip()))
            t.append(float(fields[1].strip()))
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(n, t)
    
        ax.set_xlabel("Number of live tokens")
        ax.set_ylabel(f"execution time (s)")
        
        plt.show()


if __name__ == "__main__":
    """with open("Results/pipeline_file_loading/MH01/run_1/test.dat", "r") as f:
        lines = f.readlines()
        data_pipeline = process_pipeline_times_output_file(lines, 1e3)"""
        
    foo = process_orbslam_dir("Results/baseline", "timestamp")
    #foo2 = process_orbslam_dir("Results/pipeline_file_loading", "timestamp")
    #foo3 = process_orbslam_dir("Results/full_pipeline", "timestamp")
    #foo_pip = process_orbslam_dir("Results/pipeline_file_loading", "pipeline")
    
    #compare_tasks((foo,foo2,foo3), ("baseline", "pip_file","pip_full"))
    #compare_tasks_sequential_pipeline(foo3, foo_pip, (("t_load_image",), ("t_track",)), ("Load image", "Track"))
    
    #asd = process_orbslam_dir("Results/test_rectify", "orbslam")
    #plot_timeline_tasks(foo2, 0, 2 )
    #plot_speedup([foo, foo2, foo3], ("baseline", "pip_file","pip_full"), mode='absolute')
    #plt.show()
    
    
    
    
    
    #with open("pipeline_tests/output.dat", "r") as f:
    """with open("Results/full_pipeline/MH05/run_1/PipelineTimer.dat", "r") as f:
        lines = f.readlines()
        data = process_pipeline_times_output_file(lines, 1e6, 250)
        
        draw_boxes(data, "ms")"""
        
    #foo = process_orbslam_dir("Results/baseline", "orbslam")
    #plot_tasks_bars(foo, 'datasets', version='Baseline')
    #foo2 = process_orbslam_dir("Results/pipeline_file_loading", "timestamp")
    #plot_tasks_bars(foo, 'datasets')
    #plot_speedup([foo, foo2], ["baseline", "pipeline"], mode='frame')
    #plt.show()
    
    """foo = process_orbslam_dir("Results/first_benchmark")
    
    for dataset in foo:
        runs_track = [run["t_track"] for run in dataset["runs"]]
        labels = [f"Run {i}" for i in range(len(dataset["runs"]))]
        plot_cdf_frame_time(runs_track, labels)
        plt.title(f"Different runs in {dataset['name']}")
    
   
    
    times = [[run['t_track'] for run in dataset["runs"]] for dataset in foo]
    times = [np.concatenate(dataset) for dataset in times]
    labels = [dataset['name'] for dataset in foo]
    
    plot_cdf_frame_time(times, labels, normalize=True)
    plt.title(f"All runs agregated for different datasets")
    
    plt.show()"""
    
    
    
    
    
    """foo = process_orbslam_dir("Results/first_benchmark")
    
    for dataset in foo:
        runs_track = [run["t_track"] for run in dataset["runs"]]
        labels = [f"Run {i}" for i in range(len(dataset["runs"]))]
        plot_cdf_frame_time(runs_track, labels)
        plt.title(f"Different runs in {dataset['name']}")
    
   
    
    times = [[run['t_track'] for run in dataset["runs"]] for dataset in foo]
    times = [np.concatenate(dataset) for dataset in times]
    labels = [dataset['name'] for dataset in foo]
    
    plot_cdf_frame_time(times, labels, normalize=True)
    plt.title(f"All runs agregated for different datasets")
    
    plt.show()"""
    
    
    
    
    
    
    
    
    #LOAD A FILE AND READ MANUAL TIMESTAMPS
    """with open(file_dir, "r") as f:
        lines = f.readlines()
    
    t_frame, t_load_image, t_track, t_wait, t_algo, T = process_orbslam_output_file(lines)
    
    plt.plot(t_load_image, label='LOAD')
    plt.plot(t_track, label='TRACK')
    plt.plot(t_wait, label='WAIT')
    
    
    plt.xlabel('# Frame')
    plt.ylabel("Time per frame (ms)")
    plt.legend()
    plt.show()
"""
