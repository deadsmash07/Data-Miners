import sys 
import subprocess
import matplotlib.pyplot as plt 
import time

path_apriori_executable = sys.argv[1]
path_fp_executable = sys.argv[2]
path_dataset = sys.argv[3]
path_out = sys.argv[4]

thresholds = [5,10,25,50,90]
# thresholds = [50,90]
TIMEOUT_LIMIT = 3600

times_apriori = []
times_fptree = []

for threshold in thresholds : 
    options = f'-s{threshold}'
    file_output = path_out + f'/ap{threshold}'
    command = f"{path_apriori_executable} {options} {path_dataset} {file_output}"
    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, timeout = TIMEOUT_LIMIT)
        end_time = time.time()
        times_apriori.append(end_time - start_time)
    except subprocess.TimeoutExpired : 
        times_apriori.append(TIMEOUT_LIMIT)
        with open(file_output, 'w') as f:
            pass
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        elapsed = end_time - start_time
        if "no (frequent) items found" in e.stderr:
            times_apriori.append(elapsed)
        else:
            times_apriori.append(TIMEOUT_LIMIT)
        with open(file_output, 'w') as f:
            pass
    

for threshold in thresholds : 
    options = f'-s{threshold}'
    file_output = path_out + f'/fp{threshold}'
    command = f"{path_fp_executable} {options} {path_dataset} {file_output}"

    # Run the command
    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, timeout= TIMEOUT_LIMIT)
        end_time = time.time()
        times_fptree.append(end_time - start_time)
    except subprocess.TimeoutExpired : 
        times_fptree.append(TIMEOUT_LIMIT)
        with open(file_output, 'w') as f:
            pass
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        elapsed = end_time - start_time
        # Check if the error is because no frequent items were found.
        if "no (frequent) items found" in e.stderr:
            times_fptree.append(elapsed)
        else:
            times_fptree.append(TIMEOUT_LIMIT)
        with open(file_output, 'w') as f:
            pass
# print(times_apriori)
# print(times_fptree)
plt.plot(thresholds, times_apriori, label='Apriori', color="blue", marker='o', linestyle='-')
plt.plot(thresholds, times_fptree, label='Fptree', color='red', marker='s', linestyle='-')

# Add markers at each point
plt.scatter(thresholds, times_apriori, color='blue', marker='o', label='_nolegend_')
plt.scatter(thresholds, times_fptree, color='red', marker='s', label='_nolegend_')

# Labels and title
plt.xlabel("Minimum Support Threshold (in percentage)")
plt.ylabel("Time (in seconds)")
plt.title("Execution Time vs. Minimum Support Threshold")

# Add a legend in the top right box
plt.legend(loc='upper right', title="Algorithm")

# Save the plot
plt.savefig(f'{path_out}/plot.png')