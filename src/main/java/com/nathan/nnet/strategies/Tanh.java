package com.nathan.nnet.strategies;

public class Tanh implements Strategy {
    @Override
    public Activation computeStrategy(float[] input) {

        float[] activations = new float[input.length];
        float[] derivatives = new float[input.length];

        for (int x = 0; x < input.length; x++) {
            float activation = (float) ((Math.exp(input[x]) - Math.exp(-1 * input[x])) / (Math.exp(input[x]) + Math.exp(-1 * input[x])));
            activations[x] = activation;
            derivatives[x] = (float) (1 - Math.pow(activation, 2.0));
        }

        return new Activation(activations, derivatives);
    }
}
