package com.nathan.nnet.strategies;

public interface Strategy {
    Activation computeStrategy(float[] input);
}
