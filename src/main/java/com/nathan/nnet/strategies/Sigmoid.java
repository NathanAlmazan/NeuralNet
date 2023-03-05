package com.nathan.nnet.strategies;

public class Sigmoid implements Strategy {
    @Override
    public Activation computeStrategy(float[] input) {

        float[] activations = new float[input.length];
        float[] derivatives = new float[input.length];

        for (int x = 0; x < input.length; x++) {
            float activation = (float) (1 / (1 + Math.exp(-1 * input[x])));
            activations[x] = activation;
            derivatives[x] = activation * (1 - activation);
        }

        return new Activation(activations, derivatives);
    }
}
