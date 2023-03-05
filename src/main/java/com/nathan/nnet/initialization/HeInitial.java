package com.nathan.nnet.initialization;

import java.util.Random;

public class HeInitial implements WeightInitialization {
    @Override
    public float[][] findInitialWeights(int inputSize, int outputSize) {

        Random rand = new Random();
        float[][] weights = new float[outputSize][inputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = (float) ((rand.nextFloat() - rand.nextFloat()) * Math.sqrt(2.0 / inputSize));
            }
        }

        return weights;
    }
}
