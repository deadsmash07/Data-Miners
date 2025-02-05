import sys 
import subprocess
import matplotlib.pyplot as plt 
import time

path_gspan_executable = sys.argv[1]
path_fsg_executable = sys.argv[2]
path_gaston_executable = sys.argv[3]
path_dataset = sys.argv[4]
path_out = sys.argv[5]

thresholds = [50]
# thresholds = [5, 10, 25,50,90]
  # Replace with the desired options

# Construct the command
times_gspan = []
times_fsg = []
times_gaston = []

#Change file format for gspan

#Change file format for fsg
path_gspan_dataset = "/home/baadalvm/hw1/q2/gspan_data.txt"
path_fsg_dataset = "/home/baadalvm/hw1/q2/fsg_data.txt"
path_gaston_dataset = "/home/baadalvm/hw1/q2/gaston_data.txt"

for threshold in thresholds : 
    options = f'-s {float(threshold)/100.0}'
    command = f"{path_gspan_executable} -f {path_gspan_dataset} {options} -o"
    print(command)
    # Run the command
    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        end_time = time.time()
        times_gspan.append(end_time - start_time)
        print("Command executed successfully!")
        print("Output:")
        print(result.stdout)  # Output from the command
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the command:")
        print(e.stderr)  # Error output from the command
        
# for threshold in thresholds : 
#     options = f'-s {float(threshold)}'
#     file_output = path_out + f'/fsg{threshold}'
#     command = f"{path_fsg_executable} {options} {path_fsg_dataset}"

#     # Run the command
#     try:
#         start_time = time.time()
#         result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
#         end_time = time.time()
#         times_fsg.append(end_time - start_time)
#         print("Command executed successfully!")
#         print("Output:")
#         print(result.stdout)  # Output from the command
#     except subprocess.CalledProcessError as e:
#         print("An error occurred while running the command:")
#         print(e.stderr)  # Error output from the command
        
# for threshold in thresholds : 
#     file_output = path_out + f'/gaston{threshold}'
#     command = f"{path_gaston_executable} {threshold} {path_gaston_dataset} {file_output}"
#     print(command)
#     # Run the command
#     try:
#         start_time = time.time()
#         result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
#         end_time = time.time()
#         times_gspan.append(end_time - start_time)
#         print("Command executed successfully!")
#         print("Output:")
#         print(result.stdout)  # Output from the command
#     except subprocess.CalledProcessError as e:
#         print("An error occurred while running the command:")
#         print(e.stderr)  # Error output from the command

print(times_gspan)
print(times_fsg)
print(times_gaston)
# plt.plot(thresholds,times_apriori,label='Apriori', color="blue")
# plt.plot(thresholds,times_fptree,label='Fptree', color='red')
# plt.xlabel("Threshold")
# plt.ylabel("Time")
# plt.savefig(f'{path_out}/plot.png')