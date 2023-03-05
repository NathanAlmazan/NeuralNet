package com.nathan.nnet.correction;

public class MeanSquareError implements ErrorCorrection {
    @Override
    public float computeError(float[] output, float[] expected) throws Exception {
        if (output.length != expected.length) throw new Exception("Expected output and output does not match");

        float summation = 0;
        for (int x = 0; x < output.length; x++)
            summation += Math.pow((output[x] - expected[x]), 2.0);

        return summation;
    }

    @Override
    public float[] computeErrorDerivative(float[] output, float[] expected) throws Exception {
        if (output.length != expected.length) throw new Exception("Expected output and output does not match");

        float[] gradients = new float[output.length];
        for (int x = 0; x < output.length; x++)
            gradients[x] = 2 * (output[x] - expected[x]);

        return gradients;
    }
}
