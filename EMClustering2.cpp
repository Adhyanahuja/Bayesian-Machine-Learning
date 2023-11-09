#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>

#ifndef M_PI
#define M_PI 3.1415926
#endif
using namespace std;

// Structure to represent a data point
struct DataPoint {
    double x, y,z,a;
};

// Structure to represent a cluster
struct Cluster {
    double meanX, meanY,meanZ,meanA;
    double varianceX, varianceY,varianceZ,varianceA;
    double weight;
};

// Function to read and parse CSV file into a vector of DataPoint
void readCSV(const string& filename,vector<DataPoint> &data) {
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error opening file.\n";
        return;
    }

    string line;

    // Skip the header line
    getline(file, line);

    while (getline(file, line)) {
        istringstream iss(line);
        vector<string> tokens;

        while (getline(iss, line, ',')) {
            tokens.push_back(line);
        }

        if (tokens.size() >= 6) {
            // Extract sepal width and petal length and create a DataPoint

            DataPoint point = {stod(tokens[1]), stod(tokens[2]),stod(tokens[3]),stod(tokens[4])};
            //cout<<stod(tokens[1])<<" "<<stod(tokens[2])<<" "<<stod(tokens[3])<<" "<<stod(tokens[4])<<endl;
            //cout<<endl;
            data.push_back(point);
        }
    }

    return;
}

// Function to calculate the probability of a point belonging to a cluster
double calculateProbability(const DataPoint& point, const Cluster& cluster) {
    double det = cluster.varianceX * cluster.varianceY*cluster.varianceZ*cluster.varianceA;
    double expTerm = exp(-(pow(point.x - cluster.meanX, 2) / (2 * cluster.varianceX)
                          + pow(point.y - cluster.meanY, 2) / (2 * cluster.varianceY)
                          + pow(point.z - cluster.meanZ, 2) / (2 * cluster.varianceZ)
                          + pow(point.a - cluster.meanA, 2) / (2 * cluster.varianceA)));

    return cluster.weight / (2 * M_PI * sqrt(det)) * expTerm;
}

// Function to perform the Expectation step of the EM algorithm
void expectationStep(const vector<DataPoint>& data, const vector<Cluster>& clusters, vector<vector<double>>& responsibilities) {
    for (size_t i = 0; i < data.size(); ++i) {
        double totalProbability = 0.0;
        for (size_t j = 0; j < clusters.size(); ++j) {
            responsibilities[i][j] = calculateProbability(data[i], clusters[j]);
            totalProbability += responsibilities[i][j];
        }

        // Normalize responsibilities
        for (size_t j = 0; j < clusters.size(); ++j) {
            responsibilities[i][j] /= totalProbability;
        }
    }
}

// Function to perform the Maximization step of the EM algorithm
void maximizationStep(const vector<DataPoint>& data, const vector<vector<double>>& responsibilities, vector<Cluster>& clusters) {
    for (size_t j = 0; j < clusters.size(); ++j) {
        double sumWeights = 0.0;
        double sumX = 0.0, sumY = 0.0, sumZ = 0.0, sumA = 0.0;
        double sumVarX = 0.0, sumVarY = 0.0, sumVarZ = 0.0, sumVarA = 0.0;

        for (size_t i = 0; i < data.size(); ++i) {
            sumWeights += responsibilities[i][j];
            sumX += responsibilities[i][j] * data[i].x;
            sumY += responsibilities[i][j] * data[i].y;
            sumZ += responsibilities[i][j] * data[i].z;
            sumA += responsibilities[i][j] * data[i].a;
        }

        clusters[j].weight = sumWeights / data.size();
        clusters[j].meanX = sumX / sumWeights;
        clusters[j].meanY = sumY / sumWeights;
        clusters[j].meanZ = sumZ / sumWeights;
        clusters[j].meanA = sumA / sumWeights;

        for (size_t i = 0; i < data.size(); ++i) {
            sumVarX += responsibilities[i][j] * pow(data[i].x - clusters[j].meanX, 2);
            sumVarY += responsibilities[i][j] * pow(data[i].y - clusters[j].meanY, 2);
            sumVarZ += responsibilities[i][j] * pow(data[i].z - clusters[j].meanZ, 2);
            sumVarA += responsibilities[i][j] * pow(data[i].a - clusters[j].meanA, 2);
        }

        clusters[j].varianceX = sumVarX / sumWeights;
        clusters[j].varianceY = sumVarY / sumWeights;
        clusters[j].varianceZ = sumVarZ / sumWeights;
        clusters[j].varianceA = sumVarA / sumWeights;
    }
}

// Function to initialize clusters randomly
vector<Cluster> initializeClusters(int k) {
    vector<Cluster> clusters;
    srand(static_cast<unsigned int>(time(nullptr)));

    for (int i = 0; i < k; ++i) {
        Cluster cluster;
        cluster.meanX = static_cast<double>(rand()) / RAND_MAX * 10.0;
        cluster.meanY = static_cast<double>(rand()) / RAND_MAX * 10.0;
        cluster.meanZ = static_cast<double>(rand()) / RAND_MAX * 10.0;
        cluster.meanA = static_cast<double>(rand()) / RAND_MAX * 10.0;

        cluster.varianceX = 1.0;
        cluster.varianceY = 1.0;
        cluster.varianceZ = 1.0;
        cluster.varianceA = 1.0;
        cluster.weight = 1.0 / k;

        clusters.push_back(cluster);
    }

    return clusters;
}

// Function to print cluster parameters
void printClusters(const vector<Cluster>& clusters) {
    for (size_t i = 0; i < clusters.size(); ++i) {
        cout << "Cluster " << i + 1 << ": "
             << "Mean=(" << clusters[i].meanX << ", " << clusters[i].meanY<< ", " << clusters[i].meanZ<< ", " << clusters[i].meanA << "), "
             << "Variance=(" << clusters[i].varianceX << ", " << clusters[i].varianceY << ", " << clusters[i].varianceZ<< ", " << clusters[i].varianceA<< "), "
             << "Weight=" << clusters[i].weight << endl;
    }
}
double calculateLogLikelihood(const vector<DataPoint>& data, const vector<Cluster>& clusters, const vector<vector<double>>& responsibilities) {
    double logLikelihood = 0.0;

    for (size_t i = 0; i < data.size(); ++i) {
        double pointLikelihood = 0.0;
        for (size_t j = 0; j < clusters.size(); ++j) {
            pointLikelihood += calculateProbability(data[i], clusters[j]) * clusters[j].weight;
        }
        logLikelihood += log(pointLikelihood);
    }

    return logLikelihood;
}

bool hasConverged(double prevLogLikelihood, double currentLogLikelihood, double tolerance = 1e-6) {
    return fabs(currentLogLikelihood - prevLogLikelihood) < tolerance;
}
// Function to calculate the probability of a point belonging to a cluster in two dimensions
double calculateProbabilityOfNewPoint(const DataPoint& point, const Cluster& cluster) {
    double det = cluster.varianceX * cluster.varianceY;
    double expTerm = exp(-(pow(point.x - cluster.meanX, 2) / (2 * cluster.varianceX)
                          + pow(point.y - cluster.meanY, 2) / (2 * cluster.varianceY)
                          + pow(point.z - cluster.meanZ, 2) / (2 * cluster.varianceZ)
                          + pow(point.a - cluster.meanA, 2) / (2 * cluster.varianceA)));

    return cluster.weight / pow(2 * M_PI, 1.5) * sqrt(det) * expTerm;
}
// Function to classify a new data point into one of the clusters
int classifyDataPoint(const DataPoint& point, const vector<Cluster>& clusters) {
    int bestCluster = -1;
    double maxProbability = -1.0;

    for (size_t j = 0; j < clusters.size(); ++j) {
        double probability = calculateProbabilityOfNewPoint(point, clusters[j]);
        if (probability > maxProbability) {
            maxProbability = probability;
            bestCluster = static_cast<int>(j);
        }
    }

    return bestCluster;
}
int main() {
    // Generate synthetic data
    string fileName="Iris.csv";
    vector<DataPoint> data;
    
    readCSV(fileName,data);

    // Number of clusters
    int k = 3;

    // Initialize clusters
    vector<Cluster> clusters = initializeClusters(k);

    // Number of iterations
    int maxIterations = 100;


     // EM Algorithm
    double prevLogLikelihood = -INFINITY;

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Expectation step
        vector<vector<double>> responsibilities(data.size(), vector<double>(k));
        expectationStep(data, clusters, responsibilities);

        // Maximization step
        maximizationStep(data, responsibilities, clusters);
        // Calculate log-likelihood and check for convergence
        double currentLogLikelihood = calculateLogLikelihood(data, clusters, responsibilities);

        if (hasConverged(prevLogLikelihood, currentLogLikelihood)) {
            cout << "Converged after " << iteration + 1 << " iterations.\n";
            break;
        }

        prevLogLikelihood = currentLogLikelihood;
    }

    // Print final cluster parameters
    cout << "Final Clusters:\n";
    printClusters(clusters);

    
    // New data point for classification
    DataPoint newPoint = {5,2.3,3.3,1};

    // Classify the new data point
    int predictedCluster = classifyDataPoint(newPoint, clusters);

    // Print the result
    cout << "New data point belongs to Cluster " << predictedCluster + 1 << endl;

    return 0;
}