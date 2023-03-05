package com.nathan.nnet.strategies;

public class Softmax implements Strategy {
    @Override
    public Activation computeStrategy(float[] input) {

        float[] activations = new float[input.length];
        float[] derivatives = new float[input.length];

        float summation = 0;
        for (float v : input) summation += (float) Math.exp(v);

        for (int x = 0; x < input.length; x++) {
            float probability =  (float) (Math.exp(input[x]) / summation);
            activations[x] = probability;
            derivatives[x] = probability * (1 - probability);
        }

        return new Activation(activations, derivatives);
    }
}
