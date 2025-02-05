import subprocess

cmd = [
    "../Binaries/gspan",
    "-f", "q3_datasets/Mutagenicity/preprocessed_gspan.dat",
    "-s", "0.5",
    "-o",  # output frequent subgraphs
    "-i"   # include 'x:' lines with graph IDs containing the pattern
    
]

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
