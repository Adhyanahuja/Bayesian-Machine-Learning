#include <iostream>
#include <map>
#include <vector>

using namespace std;

// Define the nodes in the BBN
struct Node {
  string name;
  vector<string> states;
  map<string, double> probabilities;
};

// Define the conditional probabilities for each node
struct ConditionalProbability {
  string parentState;
  string childState;
  double probability;
};

// Create a function to calculate the probability of a node given the evidence
double calculateProbability(Node node, map<string, string> evidence) {
  double probability = 1.0;

  for (const auto& parentState : node.states) {
    if (evidence.find(parentState) == evidence.end()) {
      continue;
    }

    string childState = evidence.at(parentState);
    double conditionalProbability = node.probabilities.at(childState + "|" + parentState);
    probability *= conditionalProbability;
  }

  return probability;
}

// Create a function to classify a patient based on their symptoms
string classifyPatient(map<string, string> symptoms) {
  // Define the nodes in the BBN
  Node feverNode = {"fever", {"true", "false"}, {{}}};
  Node headacheNode = {"headache", {"true", "false"}, {{}}};
  Node muscleAcheNode = {"muscleAche", {"true", "false"}, {{}}};

  // Define the conditional probabilities for each node
  vector<ConditionalProbability> conditionalProbabilities = {
    {"true", "true", 0.6},
    {"true", "false", 0.4},
    {"false", "true", 0.3},
    {"false", "false", 0.7}
  };

  // Populate the conditional probability table for the fever node
  for (const auto& conditionalProbability : conditionalProbabilities) {
    feverNode.probabilities.insert(make_pair(conditionalProbability.childState + "|" + conditionalProbability.parentState, conditionalProbability.probability));
  }

  // Populate the conditional probability tables for the headache and muscleAche nodes
  headacheNode.probabilities.insert(make_pair("true|fever", 0.7));
  headacheNode.probabilities.insert(make_pair("false|fever", 0.3));
  headacheNode.probabilities.insert(make_pair("true|~fever", 0.2));
  headacheNode.probabilities.insert(make_pair("false|~fever", 0.8));

  muscleAcheNode.probabilities.insert(make_pair("true|fever", 0.6));
  muscleAcheNode.probabilities.insert(make_pair("false|fever", 0.4));
  muscleAcheNode.probabilities.insert(make_pair("true|~fever", 0.1));
  muscleAcheNode.probabilities.insert(make_pair("false|~fever", 0.9));

  // Calculate the probability of fever given the symptoms
  double feverProbability = calculateProbability(feverNode, symptoms);

  // Classify the patient based on the probability of fever
  if (feverProbability > 0.5) {
    return "true";
  } else {
    return "false";
  }
}

int main() {
  // Define the symptoms
  map<string, string> symptoms = {
    {"headache", "true"},
    {"muscleAche", "false"}
  };

  // Classify the patient
  string classification = classifyPatient(symptoms);

  // Print the classification
  cout << "Patient has fever: " << classification << endl;

  return 0;
}