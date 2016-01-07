#ifndef ApproxMemorizer_HPP
#define ApproxMemorizer_HPP

#include <vector>
#include <algorithm>
#include <math.h>


namespace H2O {

using namespace std;

typedef double DATA_TYPE;

const int MAX_MINIBATCH_ITERATIONS = 1;


/**
Class ApproxMemorizer implemenmts a stochastic gradient descent variation - ADADELTA (http://arxiv.org/pdf/1212.5701v1.pdf)
A linear model is created y = theta * X

Initially the idea was to process training examples by mini batches, where the size of mini batch depends on the upper 
memory bound. Running multiple iterations for each mini batch would adjust Theta to better fit a mini batch. But, after 
multiple tests it looks like there is no improvement of using mini batches. In the final implementation the size of a mini batch 
does not affect the result.
*/
class ApproxMemorizer {
private:

    size_t m_;                  // number of model parameters (2 for our example)
    double p_;                  // decay constant
    DATA_TYPE eps_;             // This constant serves the purpose both to start off the first
                                // iteration where ∆theta = 0 and to ensure progress continues to
                                // be made even if previous updates become small
    size_t miniBatchCurSize_;   // size of a current window (batch)
    vector<pair<DATA_TYPE, DATA_TYPE> > miniBatch_; // mini batch
    bool isFirstBatch_;         // set to true for the first window, flase otherwise
    DATA_TYPE rootMeanSqrError_;
    size_t n_ ;                 // number of samples

    // Below are vectors used as a temporary variables for calculating theta delta
    vector<DATA_TYPE> eg2;  // running avarage
    vector<DATA_TYPE> eDeltaTheta2;
    vector<DATA_TYPE> deltaTheta;
    vector<DATA_TYPE> grad;
    vector<DATA_TYPE> grad2;
    vector<DATA_TYPE> temp1;
    vector<DATA_TYPE> temp2;
    vector<DATA_TYPE> temp3;


public:
    ApproxMemorizer() {
        m_ = 2;             // we have only on feature so vector X has dimention 2. First item is always set to 1.
        p_ = 0.9;           // decay constant is set 0.9 which works great for both large.data and small.data sets
        eps_ = 0.000001;    // parameter epsilon 
        theta_.resize(m_,0);  // theta has dimention 2
        rootMeanSqrError_ = 0;  
        miniBatchSize_ = 1000; // miniBatchSize_
        miniBatchCurSize_ = 0; // current size of a mini batch. Last mini batch will have smaller size
        miniBatch_.resize(miniBatchSize_, pair<DATA_TYPE, DATA_TYPE>(0,0));

        // Below are vectors used as a temporary variables for calculating theta delta
        eg2.resize(m_, 0);  
        eDeltaTheta2.resize(m_, 0);
        deltaTheta.resize(m_, 0);
        grad.resize(m_, 0);
        grad2.resize(m_, 0);
        temp1.resize(m_, 0);
        temp2.resize(m_, 0);
        temp3.resize(m_, 0);
    }

    // Ctor that sets mini batch size depending on a maxByteSize parameter 
    ApproxMemorizer(long maxByteSize) {
        m_ = 2;
        p_ = 0.9;
        eps_ = 0.000001;
        theta_.resize(m_, 0);
        rootMeanSqrError_ = 0;
        miniBatchSize_ = maxByteSize / sizeof(pair<DATA_TYPE, DATA_TYPE>);
        miniBatchCurSize_ = 0;
        miniBatch_.resize(miniBatchSize_, pair<DATA_TYPE, DATA_TYPE>(0,0));

         // Below are vectors used as a temporary variables for calculating theta delta
        eg2.resize(m_, 0);  
        eDeltaTheta2.resize(m_, 0);
        deltaTheta.resize(m_, 0);
        grad.resize(m_, 0);
        grad2.resize(m_, 0);
        temp1.resize(m_, 0);
        temp2.resize(m_, 0);
        temp3.resize(m_, 0);
    }

    // Resets mini batch size
    void setMiniBatchSize(size_t miniBatchSize) {
        if (miniBatchSize < 1 || miniBatchSize > 1000000) {
            throw "Invalid mini batch size parameter";
        }
        miniBatchSize_ = miniBatchSize;
        miniBatch_.resize(miniBatchSize_, pair<DATA_TYPE, DATA_TYPE>(0,0));
    }


    // Stores exampl values in a mini batch. When mini batch fills up trainMiniBatch is called
    void train(DATA_TYPE x, DATA_TYPE y) {

        // store example in miniBatch
        miniBatch_[miniBatchCurSize_++] = pair<DATA_TYPE, DATA_TYPE>(x,y);

        if (miniBatchCurSize_ == miniBatchSize_ ) {
            trainMiniBatch();
            isFirstBatch_ = false;
            miniBatchCurSize_ = 0;
        }
    }


    // 
    DATA_TYPE predict(double x) { 
        vector<DATA_TYPE> X(m_);
        X[0] = 1;
        X[1] = x;
        return h0(X);
    }


    // Iterate through mini batch updating Theta on each iteration
    // using ADADELTA algorithm
    void trainMiniBatch()
    {
        vector<DATA_TYPE> X(2);
        X[0] = 1;

        for (size_t i=0; i < miniBatchCurSize_; i++) {
            X[1] = miniBatch_[i].first;
            multiplyVectorNumber((h0(X) - miniBatch_[i].second), X, grad);
            power(grad, grad2, 2);
            addVectors(multiplyVectorNumber(p_, eg2, temp1), multiplyVectorNumber(1 - p_, grad2, temp2), eg2);
            deltaTheta = calculateDeltaTheta(eDeltaTheta2, grad, eg2, deltaTheta);
            eDeltaTheta2 =  addVectors(multiplyVectorNumber(p_, eDeltaTheta2, temp1), multiplyVectorNumber(1 - p_, power(deltaTheta, temp3, 2), temp2), eDeltaTheta2);
            theta_ = addVectors(theta_, deltaTheta, theta_);
        }
    }


    // 3 functions below are for calculating a mean squared error 
    void startRSME()
    {   
        rootMeanSqrError_ = 0;
        n_ = 0;
    }

    void addRSME(vector<DATA_TYPE> x, DATA_TYPE y)
    {
        rootMeanSqrError_ += pow(y - h0(x), 2);
        n_++;
    }

    DATA_TYPE getRSME()
    {
        return ::sqrt(rootMeanSqrError_ / n_);
    }


private:

    // Each item of the vector a to the power of base
    vector<DATA_TYPE>& power(const vector<DATA_TYPE>& a, vector<DATA_TYPE>& out, double base)
    {
        for (size_t i=0; i < a.size(); i++) {
            out[i] = pow(a[i], base);
        }
        return out;
    }


    // Sqrt for each item of the vector a 
    vector<DATA_TYPE>& sqrt(const vector<DATA_TYPE>& a, vector<DATA_TYPE>& out)
    {
        return power(a, out, 0.5);
    }


    // Multiply 2 vectors
    DATA_TYPE multiplyVectors(const vector<DATA_TYPE>& a, const vector<DATA_TYPE>& b) const
    {
        if (a.size() != b.size()) {
            throw "Invalid arguments";
        }

        DATA_TYPE res = 0;

        for (size_t i=0; i < a.size(); i++) {
            res += (a[i] * b[i]);
        }
        return res;
    }


    // Add 2 vectors
    vector<DATA_TYPE>& addVectors(const vector<DATA_TYPE>& a, const vector<DATA_TYPE>& b, vector<DATA_TYPE>& out) const
    {
        if (a.size() != b.size()) {
            throw "Invalid arguments";
        }
        
        for (size_t i=0; i < a.size(); i++) {
            out[i] = (a[i] + b[i]);
        }

        return out;

    }


    // Multiply each vector item to c
    vector<DATA_TYPE>& multiplyVectorNumber(DATA_TYPE c, const vector<DATA_TYPE>& v, vector<DATA_TYPE>& out) const {
        for (size_t i=0; i < v.size(); i++) {
            out[i] = (c * v[i] );
        }

        return out;
    }


    // Add c to each vector item
    vector<DATA_TYPE>& addVectorNumber(DATA_TYPE c, const vector<DATA_TYPE>& v, vector<DATA_TYPE>& out) const {

        for (size_t i=0; i < v.size(); i++) {
            out[i] = (c + v[i] );
        }

        return out;
    }


    // Calculate y value using current theta parameters
    DATA_TYPE h0(vector<DATA_TYPE> x) {
        return multiplyVectors(theta_, x);
    }


    vector<DATA_TYPE>& calculateDeltaTheta(
        const vector<DATA_TYPE>& eDeltaTheta2, 
        const vector<DATA_TYPE>& grad, 
        const vector<DATA_TYPE>& eg2,
        vector<DATA_TYPE>& out
        ) 
    {
        if (eDeltaTheta2.size() != grad.size() || eDeltaTheta2.size() != eg2.size()) {
            throw "Invalid arguments";
        }
        
        for (size_t i=0; i < eDeltaTheta2.size(); i++) {
            out[i] = -(::sqrt(eDeltaTheta2[i] + eps_) / ::sqrt(eg2[i] + eps_)) * grad[i]; 
        }
        return out;
    }


public:
    vector<DATA_TYPE> theta_;   // vector of model parameters
    size_t miniBatchSize_;      // max size of a window (batch)


};

} // Namespace H2O

#endif