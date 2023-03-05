package com.nathan.nnet.initialization;


import java.util.Random;

public class NormalRandom implements WeightInitialization {
    @Override
    public float[][] findInitialWeights(int inputSize, int outputSize) {

        Random rand = new Random();
        float[][] weights = new float[outputSize][inputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = rand.nextFloat();
            }
        }

        return weights;
    }
}
