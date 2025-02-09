#define _CRT_SECURE_NO_WARNINGS 
#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <armadillo>
#include <mlpack/core/data/split_data.hpp>
#include <string>

using namespace mlpack;
using namespace std;

// Function to load data from CSV file into an arma::mat
void loadData(const string& filePath, arma::mat& data) {
    mlpack::data::Load(filePath, data, true);
}

// Function to split the data into training and testing sets
void splitData(const arma::mat& data, arma::mat& trainData, arma::mat& testData, double testRatio) {
    mlpack::data::Split(data, trainData, testData, testRatio);
}

// Function to extract features and labels from the dataset
void extractFeaturesAndLabels(const arma::mat& data, arma::mat& X, arma::Row<size_t>& y) {
    X = data.submat(0, 0, data.n_rows - 2, data.n_cols - 1); // Features (all rows, except last column)
    y = arma::conv_to<arma::Row<size_t>>::from(data.row(data.n_rows - 1)); // Labels (last row)
}

// Function to train and evaluate the Random Forest model
void trainAndEvaluateRandomForest(const arma::mat& X_train, const arma::Row<size_t>& y_train, const arma::mat& X_test, const arma::Row<size_t>& y_test, arma::Row<size_t>& predictions) {
    // Train a RandomForest model
    mlpack::RandomForest <> rf; // Step 1: create model.
    rf.Train(X_train, y_train, 10); // Step 2: train model with 10 trees.

    // Classify points and get predictions
    rf.Classify(X_test, predictions); // Step 3: classify test points.

    // Evaluate accuracy of predictions
    double accuracy = arma::accu(predictions == y_test) / static_cast<double>(y_test.n_elem);
    cout << "Accuracy on test set: " << accuracy << endl;

    // Count zeros and ones in predictions
    size_t countZerosPredicted = arma::accu(predictions == 0);
    size_t countOnesPredicted = arma::accu(predictions == 1);

    cout << "Number of zeros predicted: " << countZerosPredicted << endl;
    cout << "Number of ones predicted: " << countOnesPredicted << endl;

    // Count actual zeros and ones in y_test
    size_t countZerosActual = arma::accu(y_test == 0);
    size_t countOnesActual = arma::accu(y_test == 1);

    cout << "Number of actual zeros in y_test: " << countZerosActual << endl;
    cout << "Number of actual ones in y_test: " << countOnesActual << endl;
    cout << arma::accu(predictions == y_test) << endl;
    cout << static_cast<double>(y_test.n_elem) << endl;
}


int main() {
    arma::mat data;
    string processedDataPath = "C:/Users/Mohand/Downloads/attack_new.csv";
    loadData(processedDataPath, data);

    // Check if data loaded successfully
    if (data.is_empty()) {
        cerr << "Error: Data loading failed or data matrix is empty." << endl;
        return 1; // Return 1 to indicate failure
    }

    // Optionally, print out some information about the loaded data
    cout << "Data loaded successfully." << endl;
    cout << "Dimensions of the loaded data: " << data.n_rows << " rows x " << data.n_cols << " columns." << endl;

    arma::mat trainData, testData;
    splitData(data, trainData, testData, 0.4);

    arma::mat X_train, X_test;
    arma::Row<size_t> y_train, y_test;
    extractFeaturesAndLabels(trainData, X_train, y_train);
    extractFeaturesAndLabels(testData, X_test, y_test);

    arma::Row<size_t> predictions;
    trainAndEvaluateRandomForest(X_train, y_train, X_test, y_test, predictions);


	

}