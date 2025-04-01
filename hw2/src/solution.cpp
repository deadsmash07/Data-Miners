#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <unordered_set>
#include <omp.h>
#include <random>

using namespace std;

struct Edge {
    int u, v;
    double p;
};

void readGraph(const string &filename, vector<Edge>& edges, int &maxNode) {
    ifstream infile(filename);
    if (!infile.is_open()){
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }
    string line;
    maxNode = 0;
    while(getline(infile, line)) {
        if(line.empty()) continue;
        istringstream iss(line);
        int u, v;
        double p;
        if (!(iss >> u >> v >> p)) continue;
        edges.push_back({u, v, p});
        maxNode = max(maxNode, max(u, v));
    }
    infile.close();
}

// Given a seed set, perform a multi-source BFS to compute the number of nodes reached.
int simulateSpread(const vector<vector<int>> &graphInstance, const unordered_set<int> &seedSet, int n) {
    vector<bool> visited(n+1, false);
    queue<int> q;
    int count = 0;
    // Initialize the queue with all seeds.
    for (int node : seedSet) {
        if (!visited[node]) {
            visited[node] = true;
            q.push(node);
            count++;
        }
    }
    // Standard BFS.
    while(!q.empty()){
        int u = q.front();
        q.pop();
        for (int v : graphInstance[u]) {
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
                count++;
            }
        }
    }
    return count;
}

int main(int argc, char* argv[]){
    if(argc != 5) {
        cerr << "Usage: " << argv[0] << " <graph_file_path> <output_file_path> <k> <#_random_instances>" << endl;
        return 1;
    }

    cout << "Total available CPUs: " << omp_get_num_procs() << endl;
    #pragma omp parallel 
    {
        #pragma omp single 
        cout << "Number of threads being used: " << omp_get_num_threads() << endl; 
    }

    string graphFile = argv[1];
    string outputFile = argv[2];
    int k = atoi(argv[3]);
    int numSimulations = atoi(argv[4]);

    // Fixed global seed for reproducibility.
    const unsigned int globalSeed = 42;

    vector<Edge> edges;
    int maxNode;
    readGraph(graphFile, edges, maxNode);
    int n = maxNode;  // assuming nodes are 1-indexed

    vector<vector<vector<int>>> graphSim(numSimulations, vector<vector<int>>(n+1));

    // Parallel generation of simulation instances using the fixed global seed
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < numSimulations; s++){
        // Create a thread-local random generator using the fixed seed
        unsigned int seed = globalSeed ^ (s + omp_get_thread_num());
        mt19937 rng(seed);
        uniform_real_distribution<double> dist(0.0, 1.0);

        for (const auto &edge : edges) {
            if (dist(rng) < edge.p) {
                graphSim[s][edge.u].push_back(edge.v);
            }
        }
    }

    ofstream outfile(outputFile, ios::out | ios::trunc);
    if (!outfile.is_open()){
        cerr << "Error: Could not open output file " << outputFile << endl;
        return 1;
    }

    // Greedy selection of k seed nodes.
    unordered_set<int> seedSet;
    vector<int> baseSpread(numSimulations, 0);

    for (int iter = 0; iter < k; iter++){
        int bestCandidate = -1;
        double bestMarginalGain = -1.0;

        // Evaluate each candidate in parallel.
        #pragma omp parallel
        {
            int localCandidate = -1;
            double localBestGain = -1.0;

            #pragma omp for nowait schedule(dynamic)
            for (int candidate = 1; candidate <= n; candidate++){
                // Skip if candidate is already in seedSet.
                if (seedSet.find(candidate) != seedSet.end())
                    continue;

                double totalMarginalGain = 0.0;
                // For each simulation instance, compute the marginal gain.
                for (int s = 0; s < numSimulations; s++){
                    // Create a temporary seed set that includes the candidate.
                    unordered_set<int> tempSeed = seedSet;
                    tempSeed.insert(candidate);
                    int spreadWithCandidate = simulateSpread(graphSim[s], tempSeed, n);
                    totalMarginalGain += (spreadWithCandidate - baseSpread[s]);
                }
                double avgMarginalGain = totalMarginalGain / numSimulations;

                // Keep track of the best candidate in this thread.
                if (avgMarginalGain > localBestGain) {
                    localBestGain = avgMarginalGain;
                    localCandidate = candidate;
                }
            }

            // Critical section to update the global best candidate.
            #pragma omp critical
            {
                if (localBestGain > bestMarginalGain) {
                    bestMarginalGain = localBestGain;
                    bestCandidate = localCandidate;
                }
            }
        } 

        if (bestCandidate == -1) break;
        seedSet.insert(bestCandidate);

        outfile << bestCandidate << "\n";
        outfile.flush();

        #pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < numSimulations; s++){
            baseSpread[s] = simulateSpread(graphSim[s], seedSet, n);
        }
        cout << "Iteration " << iter+1 << ": Selected node " << bestCandidate 
             << " with avg marginal gain " << bestMarginalGain << endl;
    }

    outfile.close();

    return 0;
}
