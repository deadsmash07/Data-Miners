import sys 
import subprocess
import matplotlib.pyplot as plt 
import time

path_apriori_executable = sys.argv[1]
path_fp_executable = sys.argv[2]
path_dataset = sys.argv[3]
path_out = sys.argv[4]

thresholds = [50]
# thresholds = [5, 10, 25,50,90]
  # Replace with the desired options

# Construct the command
times_apriori = []
times_fptree = []

for threshold in thresholds : 
    options = f'-s{threshold}'
    file_output = path_out + f'/ap{threshold}'
    command = f"{path_apriori_executable} {options} {path_dataset} {file_output}"

    # Run the command
    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        end_time = time.time()
        times_apriori.append(end_time - start_time)
        print("Command executed successfully!")
        print("Output:")
        print(result.stdout)  # Output from the command
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the command:")
        print(e.stderr)  # Error output from the command
    

for threshold in thresholds : 
    options = f'-s{threshold}'
    file_output = path_out + f'/fp{threshold}'
    command = f"{path_fp_executable} {options} {path_dataset} {file_output}"

    # Run the command
    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        end_time = time.time()
        times_fptree.append(end_time - start_time)
        print("Command executed successfully!")
        print("Output:")
        print(result.stdout)  # Output from the command
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the command:")
        print(e.stderr)  # Error output from the command
    
print(times_apriori)
print(times_fptree)
plt.plot(thresholds,times_apriori,label='Apriori', color="blue")
plt.plot(thresholds,times_fptree,label='Fptree', color='red')
plt.xlabel("Threshold")
plt.ylabel("Time")
plt.savefig(f'{path_out}/plot.png')