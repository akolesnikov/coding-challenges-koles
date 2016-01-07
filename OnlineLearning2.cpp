// OnlineLearning2.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <ostream>
#include <string>
#include <sstream>
#include <string>
#include <vector>
#include <climits>
#include <fstream>

#include "SGD.hpp"

// google's gperftools open source library for profiling
//#include "gperftools/profiler.h"


using namespace std;


int main(int argc, char* argv[])
{
    ifstream f;
    string line;
    size_t iter = 0;

    if (argc < 3) {
        throw "Usage: OnlineLearning2.exe <data file> <Mini batch size>";
    }
    f.open(argv[1]);
    if (!f.is_open()) {
        throw "Could not open the file" ;
    }

    // profiling
    //ProfilerStart("profiler1.txt");

    // Creating ApproxMemorizer object 
    // Default mini batch size is 1000
    H2O::ApproxMemorizer learning;

    // setting mini batch size to the value provided in a second argument
    learning.setMiniBatchSize(atoi(argv[2]));

    cout << "Running SDG with mini batch size set to " << learning.miniBatchSize_ << endl;

    // Reading input file and call train for each line
    while(getline(f, line)) {
        istringstream  str(line);
        H2O::DATA_TYPE x, y;
        str >> x; str >> y;
                   
        learning.train(x, y);
        iter++;
    }
    f.close();

    // train the last miniBatch
    learning.trainMiniBatch();


    // Calculate a mean squared error 
    {
        f.open(argv[1]);

        vector<H2O::DATA_TYPE> X(2);
        X[0] = 1;
        learning.startRSME();
        while(getline(f, line)) {
            istringstream  str(line);
            H2O::DATA_TYPE y;
            str >> X[1]; str >> y;
            learning.addRSME(X, y);
        }       

        f.close();
    }

    cout << "Mean Square Error = " << learning.getRSME() << endl;
    cout << "theta1 = " << learning.theta_[0] << ", theta2 = " << learning.theta_[1] << endl;

    // Profiling
    //ProfilerStop();

	return 0;
}

