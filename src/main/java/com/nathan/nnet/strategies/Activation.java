package com.nathan.nnet.strategies;

public class Activation {
    private final float[] activations;
    private final float[] derivatives;


    public Activation(float[] activations, float[] derivatives) {
        this.activations = activations;
        this.derivatives = derivatives;
    }

    public float[] getActivations() {
        return activations;
    }

    public float[] getDerivatives() {
        return derivatives;
    }
}
