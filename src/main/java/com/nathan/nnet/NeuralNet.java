package com.nathan.nnet;


import com.nathan.nnet.correction.ErrorCorrection;
import com.nathan.nnet.initialization.WeightInitialization;
import com.nathan.nnet.strategies.Strategy;

import java.util.ArrayList;
import java.util.List;

public class NeuralNet {
    private final ErrorCorrection correction;
    private final List<NLayer> layers;
    private final float learningRate;

    public NeuralNet(
            float learningRate,
            List<NLayer> layers,
            ErrorCorrection correction
    ) {
        this.correction = correction;
        this.learningRate = learningRate;
        this.layers = layers;
    }

    public NeuralNet(
            int inputSize,
            int outputSize,
            int hLayers,
            int hLayersSize,
            float learningRate,
            WeightInitialization defaultInitialization,
            WeightInitialization outputInitialization,
            Strategy defaultStrategy,
            Strategy outputStrategy,
            ErrorCorrection correction
    ) {
        this.correction = correction;
        this.learningRate = learningRate;
        this.layers = new ArrayList<>();

        if (hLayers == 0) {
            // generate input layer
            this.layers.add(new NLayer(inputSize, outputSize, null, null, learningRate, outputInitialization, outputStrategy));
        } else {
            // generate input layer
            this.layers.add(new NLayer(inputSize, hLayersSize, null, null, learningRate, defaultInitialization, defaultStrategy));

            // generate hidden layers
            for (int x = 0; x < hLayers - 1; x++)
                this.layers.add(new NLayer(hLayersSize, hLayersSize, null, null, learningRate, defaultInitialization, defaultStrategy));

            // generate output layer
            this.layers.add(new NLayer(hLayersSize, outputSize, null, null, learningRate, outputInitialization, outputStrategy));
        }
    }

    public void printWeightsAndBiases() {
        int counter = 1;
        for (NLayer layer : this.layers) {
            System.out.println("Weights " + counter);
            layer.printInfo();

            counter++;
        }
    }

    public void trainModel(List<TrainingData> trainingData, int size) throws Exception {
        for (int e = 0; e < Math.floor((double) trainingData.size() / (double) size); e++) {
            float error = 0;
            for (int t = e * size; t < ((e * size) + size ); t++) {
                TrainingData data = trainingData.get(t);
                float[] activations = data.getInputData();

                // forward propagation
                for (NLayer layer : this.layers)
                    activations = layer.computeActivation(activations);

                error += this.correction.computeError(activations, data.getOutputData());

                float[] gradients = this.correction.computeErrorDerivative(activations, data.getOutputData());

                // backward propagation
                for (int x = this.layers.size() - 1; x >= 0; x--) {
                    NLayer layer = this.layers.get(x);

                    gradients = layer.computeDerivative(gradients, x == 0);
                }
            }

            System.out.println("Epoch " + e + ": " + error);

            // update weights and biases
            for (NLayer layer : this.layers)
                layer.updateWeightsAndBiases(size);
        }
    }

    public float[] runModel(float[] inputs) throws Exception {
        float[] activations = inputs;

        // forward propagation
        for (NLayer layer : this.layers)
            activations = layer.computeActivation(activations);

        return activations;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public ErrorCorrection getCorrection() {
        return correction;
    }

    public List<NLayer> getLayers() {
        return layers;
    }
}
