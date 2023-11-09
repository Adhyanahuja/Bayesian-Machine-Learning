#include <bits/stdc++.h>
using namespace std;

class NaiveBayesClassifier {
private:
    map<string, map<string, int>> classFeatureCounts;
    map<string, int> classCounts;
    int totalDocuments;

public:
    void train(vector<vector<string>> data, vector<string> labels) {
        totalDocuments = data.size();

        for (int i = 0; i < totalDocuments; i++) {
            string label = labels[i];
            classCounts[label]++;

            for (string feature : data[i]) {
                classFeatureCounts[label][feature]++;
            }
        }
    }

    string predict(vector<string> features) {
        map<string, double> classProbabilities;

        for (auto& classEntry : classCounts) {
            string label = classEntry.first;
            double classProbability = log((double) classCounts[label] / totalDocuments);
            cout<<"Class Label: "<<label<<endl;

            for (string feature : features) {
                double featureProbability = (double) (classFeatureCounts[label][feature] + 1) / (classCounts[label] + len(classFeatureCounts[label]));
                cout<<"Feature: "<<feature<<" having probability: "<<featureProbability<<endl;
                classProbability += log(featureProbability);
            }

            classProbabilities[label] = classProbability;
            cout<<endl;
        }

        double maxProbability = (double) INT_MIN;
        string predictedLabel;

        for (auto& classProbabilityEntry : classProbabilities) {
            string label = classProbabilityEntry.first;
            double classProbability = classProbabilityEntry.second;
            cout<<"Log of probability for class "<<label<<": "<<classProbability<<endl;

            if (classProbability > maxProbability) {
                maxProbability = classProbability;
                predictedLabel = label;
            }
        }

        return predictedLabel;
    }

private:
    int len(map<string, int> m) {
        int count = 0;
        for (auto& entry : m) {
            count++;
        }
        return count;
    }
};

int main() {
    // Example usage
    vector<vector<string>> data = {
        {"sunny", "hot", "high", "false"},
        {"sunny", "cool", "high", "true"},
        {"overcast", "hot", "high", "false"},
        {"rainy", "mild", "high", "true"},
        {"rainy", "cool", "normal", "false"},
        {"rainy", "cool", "normal", "true"},
        {"overcast", "mild", "high", "true"},
        {"sunny", "mild", "normal", "false"},
        {"sunny", "cool", "normal", "true"},
        {"rainy", "mild", "normal", "false"},
        {"sunny", "mild", "high", "false"},
        {"overcast", "hot", "normal", "true"},
        {"rainy", "mild", "normal", "true"}
    };

    vector<string> labels = {
        "no", "no", "yes", "yes", "no", "yes", "yes", "no", "yes", "no", "no", "yes", "yes"
    };

    NaiveBayesClassifier classifier;
    classifier.train(data, labels);

    vector<string> newFeatures = {"overcast", "hot", "normal", "false"};
    string prediction = classifier.predict(newFeatures);
	  cout<<endl;
    cout <<"Final Class Prediction: " << prediction << endl;

    return 0;
}
