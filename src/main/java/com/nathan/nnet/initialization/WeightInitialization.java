package com.nathan.nnet.initialization;

public interface WeightInitialization {
    float[][] findInitialWeights(int inputSize, int outputSize);
}
