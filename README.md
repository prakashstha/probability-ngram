# Probability: N-gram Language Model

This was one of our assignment in CS760: Artificial Intelligence. The goal of this assignment was to give us some experience implementing a probabilistic model. It assumes you have already familiarized yourself with Git and Maven.


## Our part of the code

Our task was to implement the methods in the class `edu.uab.cis.probability.ngram.NgramLanguageModel`. 

*   `probability(List<T>)`: Return the estimated n-gram probability of the sequence, calculated as P(w<sub>0</sub>,...,w<sub>k</sub>) = ∏<sub>i=0,...,k</sub> P(w<sub>i</sub>|w<sub>i-n+1</sub>, w<sub>i-n+2</sub>, ..., w<sub>i-1</sub>).
    For example, a 3-gram language model would calculate the probability of the sequence `[A,B,B,C,A]` as `P(A,B,B,C,A) = P(A)*P(B|A)*P(B|A,B)*P(C|B,B)*P(A|B,C)`.
    *   The exact calculation of the conditional probabilities in this expression depends on the smoothing method:
        *   If the smoothing method is `NONE`, then an n-gram w<sub>1</sub>,...,w<sub>n</sub> will have its joint probability P(w<sub>1</sub>,...,w<sub>n</sub>) estimated as #(w<sub>1</sub>,...,w<sub>n</sub>) / N, where N indicates the total number of all 1-grams observed during training. Note that we have defined only the joint probability of an n-gram here. You will have to derive the conditional probability from this definition and your knowledge of probability theory.
        *   If the smoothing method is `LAPLACE`, then an n-gram w<sub>1</sub>,...,w<sub>n</sub> will have its conditional probability P(w<sub>n</sub>|w<sub>1</sub>,...,w<sub>n-1</sub>) estimated as (1 + #(w<sub>1</sub>,...,w<sub>n</sub>)) / (V + #(w<sub>1</sub>,...,w<sub>n-1</sub>)), where # indicates the number of times an n-gram was observed during training and V indicates the number of *unique* 1-grams observed during training. Note that Laplace smoothing directly defines the conditional probability of an n-gram.
    *   The calculation also depends on the representation selected:
        *   If the representation is `PROBABILITY`, then you should calculate probabilities as normal, resulting in numbers in the range `[0,1]`.
        *   If the representation is `LOG_PROBABILITY`, then you should calculate log-probabilities instead of probabilities. In every case where probabilities would have been multiplied, take advantage of the fact that `log(P(x)*P(y)) = log(P(x)) + log(P(y))` and add log-probabilities instead. This will improve efficiency since addition is faster than multiplication, and will avoid some numerical underflow problems that occur when taking the product of many small probabilities close to zero. Your log-probabilities should be in the range `(-∞,0]`.

* `train(List<T>)`: Train the language model with the n-grams from a sequence of items. Collect whatever statistics you need from the n-grams in the training sequence to be able to predict probabilities later in `probability(List<T>)`.

* `NgramLanguageModel(int, Representation, Smoothing)`: Creates an n-gram language model, specifying the representation and smoothing to be used by `probability(List<T>)`.

## Compile and test the code

1.  Compile the code. Run the following command:

        mvn clean compile

    Everything should compile and you should see a message like:

        [INFO] ------------------------------------------------------------------------
        [INFO] BUILD SUCCESS
        [INFO] ------------------------------------------------------------------------

2.  Test the code. Run the following command:

        mvn clean test

    You should now see a message like:

        [INFO] ------------------------------------------------------------------------
        [INFO] BUILD SUCCESS
        [INFO] ------------------------------------------------------------------------