package com.nathan.nnet.correction;

public class CrossEntropy implements ErrorCorrection {
    @Override
    public float computeError(float[] output, float[] expected) throws Exception {
        if (output.length != expected.length) throw new Exception("Expected output and output does not match");

        float summation = 0;
        for (int x = 0; x < output.length; x++) {
            if (expected[x] == 1) summation += (float) -1 * Math.log(output[x]);
        }

        return summation;
    }

    @Override
    public float[] computeErrorDerivative(float[] output, float[] expected) throws Exception {
        if (output.length != expected.length) throw new Exception("Expected output and output does not match");

        float[] gradients = new float[output.length];
        for (int x = 0; x < output.length; x++)
            gradients[x] = output[x] - expected[x];

        return gradients;
    }
}
