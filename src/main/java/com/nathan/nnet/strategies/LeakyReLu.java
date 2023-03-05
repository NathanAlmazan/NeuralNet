package com.nathan.nnet.strategies;

public class LeakyReLu implements Strategy {
    @Override
    public Activation computeStrategy(float[] input) {

        float[] activations = new float[input.length];
        float[] derivatives = new float[input.length];

        for (int x = 0; x < input.length; x++) {
            if (input[x] > 0) activations[x] = input[x];
            else activations[x] = input[x] * 0.1f;

            if (input[x] > 0) derivatives[x] = 1;
            else derivatives[x] = 0.1f;
        }

        return new Activation(activations, derivatives);
    }
}
