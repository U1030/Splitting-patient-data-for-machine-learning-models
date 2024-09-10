from multiprocessing import Pool
from functools import partial
import time
from tqdm import tqdm
import psutil

def parallelize(data, function, num_processes=-1):
    if num_processes <= 0:
        num_cpus = psutil.cpu_count(logical=False)
    else:
        num_cpus = num_processes
    pool = Pool(processes=num_cpus)
    data_list = list(data) 
    results = list(tqdm(pool.imap_unordered(function, data_list), total=len(data_list)))
    pool.close()
    pool.join()
    return results

def parallelize_maintain_index(data, function, num_processes=-1):
    if num_processes <= 0:
        num_cpus = psutil.cpu_count(logical=False)
    else:
        num_cpus = num_processes   
    with Pool(processes=num_cpus) as pool:
        results_dict = {}
        for index, row_data in enumerate(data):
            results_dict[index] = pool.apply_async(function, args=(row_data,))        
        pool.close()
        pool.join()        
        results = [results_dict[index].get() for index in sorted(results_dict)]    
    return results

def parralelize__several_args(data, function, processes=-1, **kwargs):  
    start = time.time()    
    if processes <= 0:
        num_cpus = psutil.cpu_count(logical=False)
    else:
        num_cpus = processes
    process_pool = Pool(processes=num_cpus)
    data_list = list(data)
    results = []
    for _ in tqdm(process_pool.imap_unordered(partial(function, **kwargs), data_list), total=len(data_list)):
        results.append(_)
    process_pool.close()
    process_pool.join()
    end = time.time()
    print('Completed in: %s sec' % (end - start))
    return results
