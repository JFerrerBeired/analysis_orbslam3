import matplotlib.pyplot as plt
import numpy as np
import re
import os
from collections import defaultdict

from plotutils import *

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
            analysis_file = "PipelineTimer.dat"
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
                        sub_dict = process_pipeline_times_output_file(lines, 1e6)
                    
                    if analysis != 'pipeline':
                        file_path = os.path.join(directory, dataset, run, "SessionInfo.txt")
                        print("\t", file_path)
                        with open(file_path, "r") as f_info:
                            lines_info = f_info.readlines()
                            
                            #Compatibility for OLD versions
                            if len(lines_info) > 3:
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
                print(f"\t{fieldh}:crazy value")
            data_dict[fieldh].append(val)
    
    for fieldh in data_dict.keys():
        data_dict[fieldh] = np.nan_to_num(np.array(data_dict[fieldh]), nan=np.nan) #Convert nans to np.nan to correctly ignore them in means
    
    return data_dict
            

def process_pipeline_times_output_file(lines, time_factor_reduction=1, max_items=np.inf, max_time=None):
    #If max_time is given, max_items is ignored
    
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
        
        if max_time is not None:
            if t_start > max_time:
                continue #Skip if over max_time
        elif item >= n_items:
            continue #Skip if over max_items
        
        
        t_start /= time_factor_reduction
        t_end /= time_factor_reduction
        
        data[item][stage] = (t_start, t_end-t_start)
    
    if max_time is not None: #Filter uncompleted items
        i=0
        for d in data:
            broken = False
            for s in d:
                if s == -1:
                    broken=True
                    break
            if broken:
                break
            i+=1
        data = data[:i]
        
    
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


def process_number_tokens_pipeline(directory):
    """Expected path is number_token/dataset/run/
    
    Takes the total time via the START and END timers on orbslam3_output.log
    Takes the tasks total time via TrackingTimeStats.txt
    The output dictionary has the structure: 
        dic[dataset][task] is an array of times correspondent to number of tokens kept in array dict[n_tokens]
    It is assumed that all the subfolders have the same number of tokens and datasets computed
    """
    data_dict = defaultdict(lambda: defaultdict(lambda: dict()))
    n_tokens = list()
    for n_token in os.listdir(directory):
       
        for dataset in os.listdir(os.path.join(directory, n_token)):
            execution_times = list()
            algorithm_times = list()
            for run in os.listdir(os.path.join(directory, n_token, dataset)):
                file_output = os.path.join(directory, n_token, dataset, run, 'orbslam3_output.log')
                file_tracking = os.path.join(directory, n_token, dataset, run, 'TrackingTimeStats.txt')
                
                with open(file_output, "r") as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        fields = line.split('\t')
                        if fields[0] == 'START':
                            start = int(fields[1])
                        elif fields[0] == 'ALGO_START':
                            t1 = int(fields[1])                            
                        elif fields[0] == 'ALGO_END':
                            t2 = int(fields[1])
                        elif fields[0] == 'END':
                            end = int(fields[1])
                    
                    execution_time = (end - start) /  1e9 # ns to s
                    algorithm_time = (t2 - t1) /  1e9 # ns to s
                execution_times.append(execution_time)
                algorithm_times.append(algorithm_time)
                """with open(file_tracking, "r") as f:
                    lines = f.readlines()
                    
                    data_dict = process_tracking_time_stats(lines)
                    
                    for k,v in data_dict.items(): #I want total time spent per task, not by frame
                        data_dict[k] = np.nansum(v)/1e3 # ms to s
                    task_time = np.sum(list(data_dict.values())) - data_dict['Total'] #Time spent in registered tasks
                    data_dict['TotalTracking'] = data_dict['Total']
                    data_dict['Total'] = execution_time
                    data_dict['OtherTracking'] = data_dict['TotalTracking'] - task_time
                    data_dict['Other'] = data_dict['Total'] - data_dict['TotalTracking']
                    
                    for k,v in data_dict:
                        print(k, "\t", v)
                    
                    3"""
            data_dict[dataset]["Total"][int(n_token)] = execution_times
            data_dict[dataset]["Algo"][int(n_token)] = algorithm_time

    return data_dict


if __name__ == "__main__":
    # base_data = process_orbslam_dir("Results/baseline", "timestamp")
    # token_data_parallel = process_number_tokens_pipeline("Results/pipeline_token_parallel")
    # # token_data_thread = process_number_tokens_pipeline("Results/pipeline_token_thread")
    
    # plot_number_tokens_pipeline((token_data_parallel,), ("",), base_data, task="Algo", mode='datasets')

    base_data = process_orbslam_dir("Results/baseline", "timestamp")
    d = [base_data]
    for n in (5, 10, 15, 24, 30):
        # with open(f"Results/pipeline_token_parallel/{n}/MH01/run_2/PipelineTimer.dat", "r") as f:
        #     pip_data = process_pipeline_times_output_file(f.readlines(), 1e6, max_time=1e9)
           
        # draw_boxes(pip_data, "ms", ("Read File", "Extract Features", "Track"), f"{n} pipeline tokens")
        
        d.append(process_orbslam_dir(f"Results/pipeline_token_parallel_old/{n}", "timestamp"))
    plot_speedup(d, ("b", "5", "10", "15", "24", "30"), mode='error')
    plot_speedup(d, ("b", "5", "10", "15", "24", "30"), mode='frequency')
    
    # #IDEA: Measure time last stage vs total algo time
    # """    for n in (5, 10, 15, 24, 30):
    #     pip_data = process_orbslam_dir(f"Results/pipeline_token_parallel/{n}", "pipeline")[0]["runs"][0]

    #     Ttrack = np.sum([i[2][1] for i in pip_data])
    #     time_data = process_orbslam_dir(f"Results/pipeline_token_parallel/{n}", "timestamp")[0]["runs"][0]["t_algo"]
        
    #     print(time_data-Ttrack)"""
    
    # plot_tasks_bars(base_data, 'datasets', version='Baseline')

    # #compare_tasks(d, ("base", "5", "10", "15", "24"))
    # plot_histogram_tasks(base_data)
    # plt.show()
    
    # base_data = process_orbslam_dir("Results/baseline", "timestamp")
    # base_data2 = process_orbslam_dir("Results/pipeline_token_parallel_old/5", "timestamp")
    # base_data3 = process_orbslam_dir("Results/pipeline_token_parallel_old/20", "orbslam")
    # pip_data = process_orbslam_dir("Results/full_pipeline", "orbslam")
    # pip_data_thread = process_orbslam_dir("Results/full_pipeline_thread", "orbslam")
    
    """ with open(f"Results/new_pipeline2/MH01/run_1/PipelineTimer.dat", "r") as f:
        pip_data = process_pipeline_times_output_file(f.readlines(), 1e6, max_time=1e9)
        
    draw_boxes(pip_data, "ms", ("Read File", "Exctract Features", "Track"), f"{15} pipeline tokens")"""
    
    # compare_tasks((base_data,base_data2,base_data3), ("Baseline", "Pipeline1", "Pipeline2"))
    
    #Show tasks times
    # plot_tasks_bars(base_data, 'datasets', version='Baseline')
    # plot_histogram_tasks(base_data)
    # plot_speedup((base_data, base_data2), ("1", "2"), mode='error')

    # plot_tasks_bars(base_data2, 'datasets', version='Pipeline')
    #plot_tasks_bars(base_data, 'runs', version='Baseline')
    
    # plot_tasks_bars(pip_data, 'datasets', version='Pipeline')
    # plot_tasks_bars(pip_data, 'runs', version='Pipeline')
    
    # compare_tasks((base_data,pip_data,pip_data_thread), ("Baseline", "Pipeline", "Pipeline_threads"))
    
    # pip_data = process_orbslam_dir("Results/full_pipeline", "pipeline")
    # compare_tasks_sequential_pipeline(base_data, pip_data, (("Load File",), 
    #                                                         ("Image Rect","ORB ext","Stereo match"),
    #                                                         ("Pose pred", "LM track", "KF dec")),
    #                                                         ("Load image", "Process Image", "Track"))

    
    # base_data = process_orbslam_dir("Results/baseline", "timestamp")
    # pip_data = process_orbslam_dir("Results/full_pipeline", "timestamp")
    # pip_thread = process_orbslam_dir("Results/full_pipeline_thread", "timestamp")
    # plot_speedup([base_data, pip_data,pip_thread], ("Baseline", "Pipeline", "t"), mode='absolute')
    # plot_speedup([base_data, pip_data,pip_thread], ("Baseline", "Pipeline","t"), mode='frame')
    # plot_speedup([base_data, pip_data,pip_thread], ("Baseline", "Pipeline","t"), mode='speedup')

    # with open("Results/full_pipeline/MH01/run_1/PipelineTimer.dat", "r") as f:
    #     lines = f.readlines()
    #     data = process_pipeline_times_output_file(lines, 1e6, 250)
        
    #     draw_boxes(data, "ms")
    
    
    plt.show()
