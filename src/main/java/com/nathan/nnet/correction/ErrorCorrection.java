package com.nathan.nnet.correction;

public interface ErrorCorrection {
    float computeError(float[] output, float[] expected) throws Exception;
    float[] computeErrorDerivative(float[] output, float[] expected) throws Exception;
}
