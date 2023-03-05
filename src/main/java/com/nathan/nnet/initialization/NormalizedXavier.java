package com.nathan.nnet.initialization;

import java.util.Random;

public class NormalizedXavier implements WeightInitialization {
    @Override
    public float[][] findInitialWeights(int inputSize, int outputSize) {

        double mean = Math.sqrt(6.0) / Math.sqrt(inputSize + outputSize);
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
