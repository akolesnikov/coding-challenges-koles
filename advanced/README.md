# Programming Code Challenge for Advanced Programmers

## HowTo

* Please read the [general instructions](https://github.com/h2oai/coding-challenges/blob/master/README.md)

## Problem Statement
* Write a class `ApproxMemorizer` that attempts to approximately memorize a signal of the form `y_i = f(x_i)`, given `N` observations `{x_i, y_i}` that consist of two real-valued numbers
* `N` is potentially very large (many millions or billions)
* There is no a priori knowledge of the underlying function `f`. You can assume that it's *reasonably smooth*, even though the observations might be noisy (especially for large numbers of observations). The goal is to be as generic as possible.
* The upper bound on the memory usage of the ApproxMemorizer is limited by a parameter during construction
* Training of the ApproxMemorizer class must be done one observation at a time (online learning), in one pass over the data
* After training is done, quantify the accuracy of the ApproxMemorizer with the *Mean Square Error* metric by doing another pass over the training data and comparing the actual y values with the predicted y values.

## Implementation
* You must use C++ or Java for the main implementation
* You can use tools and languages of your choice for testing, benchmarking and plotting
* Implement the following API:

        /**
         * Constructor of the ApproxMemorizer
         * @param maxByteSize upper bound for total internal data structure size in bytes
         */
        ApproxMemorizer(long maxByteSize) { ... }
        
        /**
         * Train the memorizer one observation at a time
         * @param x real-valued ordinate
         * @param y real-valued co-ordinate, where y = f(x) for some fixed but unknown f
         */
        void train(double x, double y) { ... }
        
        /**
         * Making predictions using the current internal representation of f
         * @param x real-valued ordinate
         * returns predicted real-valued co-ordinate y = f(x)
         */
        double predict(double x) { ... }
     

## Testing and Benchmarking
* Write unit tests to check that your implementation is correct (e.g., using the file `small.data`)
* Benchmark your code on the given dataset `large.data.gz` in this repository with `{x_i, y_i}` values and report the accuracy and runtime for various values of maxByteSize by creating PNG plots
* Note: We might test your solution on other datasets (do not over-optimize your solution for this dataset)

## Tips
* If you are unsure of the quality of your solution, write the testing and benchmarking code first
* Write simple code first, optimize later
* Iterate, iterate, iterate (Use git to document the evolution of your thought process)
