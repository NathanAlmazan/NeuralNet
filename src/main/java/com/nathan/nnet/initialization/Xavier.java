package com.nathan.nnet.initialization;

import java.util.Random;

public class Xavier implements WeightInitialization {
    @Override
    public float[][] findInitialWeights(int inputSize, int outputSize) {

        double mean = 1.0 / Math.sqrt(inputSize);
        float lower = (float) ( -1 * mean);
        float upper = (float) mean;

        Random rand = new Random();

        float[][] weights = new float[outputSize][inputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = lower + rand.nextFloat() * (upper - lower);
            }
        }

        return weights;
    }
}
