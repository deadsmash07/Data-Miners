import sys 
import subprocess
import matplotlib.pyplot as plt 
import time
from math import ceil
import os, shutil
import signal

path_dataset = sys.argv[1]
path_gaston_dataset = sys.argv[2]
TIMEOUT_LIMIT = 3600.0
# TIMEOUT_LIMIT = 2.0

def convert_to_gaston_format(input_file, output_file):
  cnt =0
  with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    lines = infile.readlines()
    
    index = 0
    while index < len(lines):
      if lines[index].startswith('#'):
        cnt+=1
        graph_id = lines[index].strip()[1:]  # Extracting graph ID
        index += 1
        num_nodes = int(lines[index].strip())  # Number of nodes
        index += 1
        
        # Read node labels
        node_labels = []
        for _ in range(num_nodes):
          node_labels.append(lines[index].strip())
          index += 1
        
        num_edges = int(lines[index].strip())  # Number of edges
        index += 1
        edges = []
        
        # Read edges
        for _ in range(num_edges):
          src, dest, label = map(int, lines[index].strip().split())
          edges.append((min(src, dest), max(src, dest), label))
          index += 1
        
        # Sort edges by (ID1, ID2)
        edges.sort()
        
        # Write to output file
        outfile.write(f't # {graph_id}\n')
        
        label_to_int = {'Br':0, 'C':1, 'Cl':2, 'F':3, 'H':4, 'I':5, 'N':6, 'O':7, 'P':8, 'S':9, 'Si':10}
        
        for node_id, label in enumerate(node_labels):
          outfile.write(f'v {node_id} {label_to_int[label]}\n')
        for src, dest, label in edges:
          outfile.write(f'e {src} {dest} {label}\n')
            
      else:
        index+=1
    
    infile.close() 
      
                    
  return cnt

def convert_to_gspan_format(input_file, output_file):

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        
        index = 0
        while index < len(lines):
            if lines[index].startswith('#'):
                graph_id = lines[index].strip()[1:]  # Extracting graph ID
                index += 1
                num_nodes = int(lines[index].strip())  # Number of nodes
                index += 1
                
                # Read node labels
                node_labels = []
                for _ in range(num_nodes):
                    node_labels.append(lines[index].strip())
                    index += 1

                num_edges = int(lines[index].strip())  # Number of edges
                index += 1
                edges = []
                
                # Read edges
                for _ in range(num_edges):
                    src, dest, label = map(int, lines[index].strip().split())
                    edges.append((min(src, dest), max(src, dest), label))
                    index += 1

                # Sort edges by (ID1, ID2)
                edges.sort()
                
                # Write to output file
                outfile.write(f't # {graph_id}\n')
                
                label_to_int = {'Br':0, 'C':1, 'Cl':2, 'F':3, 'H':4, 'I':5, 'N':6, 'O':7, 'P':8, 'S':9, 'Si':10}

                for node_id, label in enumerate(node_labels):
                    outfile.write(f'v {node_id} {label_to_int[label]}\n')
                for src, dest, label in edges:
                    outfile.write(f'e {src} {dest} {label}\n')
            else: 
                index+=1

def convert_to_fsg_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        
        index = 0
        while index < len(lines):
            if lines[index].startswith('#'):
                graph_id = lines[index].strip()[1:]  # Extracting graph ID
                index += 1
                num_nodes = int(lines[index].strip())  # Number of nodes
                index += 1
                
                # Read node labels
                node_labels = []
                for _ in range(num_nodes):
                    node_labels.append(lines[index].strip())
                    index += 1
                
                num_edges = int(lines[index].strip())  # Number of edges
                index += 1
                edges = []
                
                # Read edges
                for _ in range(num_edges):
                    src, dest, label = map(int, lines[index].strip().split())
                    edges.append((min(src, dest), max(src, dest), label))
                    index += 1
                
                # Sort edges by (ID1, ID2)
                edges.sort()
                
                # Write to output file
                outfile.write(f't\n')
                for node_id, label in enumerate(node_labels):
                    outfile.write(f'v {node_id} {label}\n')
                for src, dest, label in edges:
                    outfile.write(f'u {src} {dest} {label}\n')
            else:
                index+=1

# def run_tool(command, tool_name):
#     """Execute a subprocess with 60-minute timeout"""
#     try:
#         start_time = time.time()
        
#         # Launch process with separate stdout/stderr pipes
#         process = subprocess.run(
#             command,
#             shell=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#             timeout = TIMEOUT_LIMIT
#         )
        
#         elapsed = time.time() - start_time
        
#         if process.returncode != 0:
#             # print(f"{tool_name} failed (exit {process.returncode})")
#             # print("Error output:")
#             # print(stderr)
#             return TIMEOUT_LIMIT

#         return elapsed
            
#     except subprocess.TimeoutExpired:
#         return TIMEOUT_LIMIT # Return 60 mins as runtime
            
    # except Exception as e:
    #     # print(f"Error running {tool_name}: {str(e)}")
    #     return TIMEOUT_LIMIT

def run_tool(command, tool_name):
    try:
        start_time = time.time()
        
        # Start the process in a new process group
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # This creates a new process group
        )
        
        try:
            stdout, stderr = process.communicate(timeout=TIMEOUT_LIMIT)
            elapsed = time.time() - start_time
            if process.returncode != 0:
                return TIMEOUT_LIMIT
            return elapsed
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.kill()  # In case SIGTERM didn't work fast enough
            return TIMEOUT_LIMIT
    except Exception as e:
        # Optionally log the error: print(f"Error running {tool_name}: {str(e)}")
        return TIMEOUT_LIMIT
      
def main():
  if len(sys.argv) != 6:
    print("Usage: python3 run.py <gspan_exe> <fsg_exe> <gaston_exe> <dataset> <output_dir>")
    sys.exit(1)
  
  gspan_exe, fsg_exe, gaston_exe, dataset, output_dir = sys.argv[1:]
  thresholds = [95, 50, 25, 10, 5]
  # thresholds = [ 5]
  timings = {'gspan': [], 'fsg': [], 'gaston': []}
  
  current_dir = os.path.dirname(os.path.abspath(__file__))
  
  os.makedirs(output_dir, exist_ok=True)
  
  # Run Gspan experiments
  gspan_dataset = os.path.join(current_dir, "gspan_dataset.txt")
  convert_to_gspan_format(dataset, gspan_dataset)
  
  for thresh in thresholds:
    fraction_thresh = float(thresh)/100.0
    output_file = f"{output_dir}/gspan{thresh}"
    command = f"{gspan_exe} -f {gspan_dataset} -s {fraction_thresh} -o"
    # print(command)
    runtime = run_tool(command, f"Gspan ({thresh}%)")
    if runtime is not None:
      # print(f"gspan{thresh}: {runtime}")
      timings['gspan'].append(runtime)

    # Source file path
    source_file = gspan_dataset+'.fp'
    if os.path.exists(source_file) and runtime<TIMEOUT_LIMIT:
      shutil.move(source_file, output_file)
    else:
      with open(output_file, "w") as f:
        pass
      
  
  # Run Gaston experiments
  gaston_dataset = os.path.join(current_dir, "gaston_dataset.txt")
  num_graphs = convert_to_gaston_format(dataset, gaston_dataset)
  
  for thresh in thresholds:
    abs_thresh = ceil(thresh * num_graphs / 100)
    output_file = f"{output_dir}/gaston{thresh}"
    command = f"{gaston_exe} {abs_thresh} {gaston_dataset} {output_file}"
    # print(command)
    runtime = run_tool(command, f"Gaston ({thresh}%)")
    if runtime is not None :
      # print(f"gaston{thresh}: {runtime}")
      timings['gaston'].append(runtime)
    # print(runtime)
    # print(TIMEOUT_LIMIT)
    if runtime >= TIMEOUT_LIMIT:
      # print('here')
      # print(output_file)
      with open(output_file, "w") as f:
        # print('here2')
        # f.truncate(0)
        # f.write("")
        pass

  # Run FSG experiments
  fsg_dataset = os.path.join(current_dir, "fsg_dataset.txt")
  convert_to_fsg_format(dataset, fsg_dataset)
  
  for thresh in thresholds:
    float_thresh = float(thresh)
    output_file = f"{output_dir}/fsg{thresh}"
    command = f"{fsg_exe} -s {float_thresh} {fsg_dataset}"
    # print(command)
    runtime = run_tool(command, f"FSG ({thresh}%)")
    if runtime is not None:
      # print(f"fsg{thresh}: {runtime}")
      timings['fsg'].append(runtime)
      
    source_file = fsg_dataset[:-4]+'.fp'
    if os.path.exists(source_file) and runtime<TIMEOUT_LIMIT:
      shutil.move(source_file, output_file)
    else:
      with open(output_file, "w") as f:
        pass
      
  # print("fsg done")
  # print(timings)
  
  # plot the results
  # timings = {'gspan': [4.6360554695129395, 91.84061479568481, 261.35330986976624, 1298.4060633182526, 10.0], 'fsg': [20.124086141586304, 116.42788934707642, 338.32729983329773, 1180.416169166565, 3570.337596178055], 'gaston': [0.8407857418060303, 7.434269189834595, 16.23651123046875, 48.32618808746338, 138.98323273658752]}
  plt.figure(figsize=(12, 8), dpi=300)
  plt.plot(thresholds, timings['gaston'], label='Gaston', marker='o', color='blue',
         markersize=8, linewidth=2)
  plt.plot(thresholds, timings['gspan'], label='Gspan', marker='s', color='red',
          markersize=8, linewidth=2)
  plt.plot(thresholds, timings['fsg'], label='FSG', marker='^', color='green',
          markersize=8, linewidth=2)

  # Set the labels and title with increased font sizes for better readability
  plt.xlabel("Minimum Support Threshold (%)", fontsize=14)
  plt.ylabel("Time (in seconds)", fontsize=14)
  plt.title("Algorithm Timing Comparison", fontsize=16)
  
  plt.legend(fontsize=12)
  # Enable grid lines with a dashed style for easier visual tracing
  plt.grid(True, linestyle='--', alpha=0.7)

  # Adjust tick label font sizes
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)

  # Ensure a tight layout so labels and titles do not get clipped
  plt.tight_layout()

  # Save the figure to the output directory with a high DPI and tight bounding box
  plt.savefig(f'{output_dir}/plot.png', dpi=300, bbox_inches='tight')

  # Optionally display the plot interactively
  plt.show()

  
if __name__ == "__main__":
  main()