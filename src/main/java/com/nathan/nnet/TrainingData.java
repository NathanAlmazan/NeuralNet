package com.nathan.nnet;

public class TrainingData {
    private final float[] inputData;
    private final float[] outputData;


    public TrainingData(float[] inputData, float[] outputData) {
        this.inputData = inputData;
        this.outputData = outputData;
    }

    public float[] getInputData() {
        return inputData;
    }

    public float[] getOutputData() {
        return outputData;
    }
}
