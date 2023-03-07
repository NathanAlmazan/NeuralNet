package com.nathan.nnet;


import com.nathan.nnet.initialization.WeightInitialization;
import com.nathan.nnet.strategies.Activation;
import com.nathan.nnet.strategies.Strategy;


public class NLayer {
    private float[] input;
    private float[] output;
    private final int inputSize;
    private final int outputSize;
    private final float learningRate;
    private final float[] biases;
    private final float[] biasesGradient;
    private final float[][] weights;
    private final float[][] weightGradients;
    private final Strategy strategy;
    private final WeightInitialization initialization;

    public NLayer(
            int inputSize,
            int outputSize,
            float[][] weights,
            float[] biases,
            float learningRate,
            WeightInitialization initialization,
            Strategy strategy
    ) {
        this.strategy = strategy;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.initialization = initialization;

        // initialize weights, biases and weight gradients
        this.biases = new float[outputSize];
        this.biasesGradient = new float[outputSize];
        this.weightGradients = new float[outputSize][inputSize];

        // initialize weights
        if (weights == null) this.weights = initialization.findInitialWeights(inputSize, outputSize);
        else this.weights = new float[outputSize][inputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                // assign weights
                if (weights != null) this.weights[i][j] = weights[i][j];

                this.weightGradients[i][j] = 0;
            }

            // assign biases
            if (biases != null) this.biases[i] = biases[i];
            else this.biases[i] = 0;

            this.biasesGradient[i] = 0;
        }
    }

    public float[] computeActivation(float[] inputs) throws Exception {
        if (inputs.length != inputSize) throw new Exception("Input size is insufficient");
        this.input = inputs; // store input for derivatives

        // matrix multiplication
        float[] activations = new float[outputSize];
        for (int x = 0; x < this.outputSize; x++) {

            // summation of weights * input
            float summation = 0;
            for (int j = 0; j < this.inputSize; j++) summation += this.weights[x][j] * inputs[j];

            activations[x] = summation + this.biases[x]; // store summation for activation
        }

        Activation result = strategy.computeStrategy(activations);
        this.output = result.getDerivatives(); // store output for derivatives

        return result.getActivations();
    }

    public float[] computeDerivative(float[] errorCost, boolean last) throws Exception {
        if (this.output == null || errorCost.length != outputSize || this.output.length != outputSize)
            throw new Exception("Output size is insufficient");

        if (this.input == null || this.input.length != inputSize)
            throw new Exception("Input size is insufficient");

        // compute weights gradient and store weight gradients
        for (int i = 0; i < this.outputSize; i++) {
            for (int j = 0; j < this.inputSize; j++)
                this.weightGradients[i][j] += this.learningRate * this.input[j] * this.output[i] * errorCost[i];

            this.biasesGradient[i] += this.learningRate * this.output[i] * errorCost[i];
        }

        if (last) return null;

        // compute input layer derivative
        float[] inputDerivatives = new float[inputSize];
        for (int i = 0; i < this.inputSize; i++) {
            float summation = 0;
            for (int j = 0; j < this.outputSize; j++)
                summation += this.weights[j][i] * this.output[j] * errorCost[j];

            inputDerivatives[i] = summation;
        }

        return inputDerivatives;
    }

    public void updateWeightsAndBiases(int iteration) {
        // update weights
        for (int i = 0; i < this.outputSize; i++) {
            for (int j = 0; j < this.inputSize; j++) {
                this.weights[i][j] -= this.weightGradients[i][j] / iteration;
                this.weightGradients[i][j] = 0;
            }

            this.biases[i] -= this.biasesGradient[i] / iteration;
            this.biasesGradient[i] = 0;
        }
    }

    public void printInfo() {
        // print weights and biases
        for (int x = 0; x < this.output.length; x++) {
            for (int j = 0; j < this.input.length; j++)
                System.out.printf(" %.8f ", this.weights[x][j]);

            System.out.printf(" + %.8f ", this.biases[x]);
            System.out.println();
        }
        System.out.println();
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public float[] getBiases() {
        return biases;
    }

    public float[][] getWeights() {
        return weights;
    }

    public Strategy getStrategy() {
        return strategy;
    }

    public WeightInitialization getInitialization() {
        return initialization;
    }
}
