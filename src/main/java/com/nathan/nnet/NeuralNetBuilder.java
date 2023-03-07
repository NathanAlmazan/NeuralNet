package com.nathan.nnet;

import com.nathan.nnet.correction.ErrorCorrection;
import com.nathan.nnet.initialization.WeightInitialization;
import com.nathan.nnet.store.ModelStorage;
import com.nathan.nnet.strategies.Strategy;

import java.util.Arrays;
import java.util.List;

public class NeuralNetBuilder {
	private int inputSize;
	private int outputSize;
	private int hLayers;
	private int hLayersSize;
	private float learningRate;
	private WeightInitialization defaultInitialization;
	private WeightInitialization outputInitialization;
	private Strategy defaultStrategy;
	private Strategy outputStrategy;
	private ErrorCorrection correction;
	private NeuralNet neuralNet;
	private List<TrainingData> trainingData;
	private int batchSize;
	private int testSize;
	private String storageLocation;
	private ModelStorage modelStorage;

	public NeuralNetBuilder setDimensions(int inputSize, int outputSize) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;

		return this;
	}

	public NeuralNetBuilder setHiddenLayers(int hLayers, int hLayersSize) {
		this.hLayers = hLayers;
		this.hLayersSize = hLayersSize;

		return this;
	}

	public NeuralNetBuilder setWeightInitialization(WeightInitialization defaultInitialization, WeightInitialization outputInitialization) {
		this.defaultInitialization = defaultInitialization;
		this.outputInitialization = outputInitialization;

		return this;
	}

	public NeuralNetBuilder setLearningStrategy(Strategy defaultStrategy, Strategy outputStrategy) {
		this.defaultStrategy = defaultStrategy;
		this.outputStrategy = outputStrategy;

		return this;
	}

	public NeuralNetBuilder setErrorCorrection(float learningRate, ErrorCorrection correction) {
		this.learningRate = learningRate;
		this.correction = correction;

		return this;
	}

	public NeuralNetBuilder setTrainingParameters(List<TrainingData> trainingData, int batchSize, int testSize) {
		this.trainingData = trainingData;
		this.batchSize = batchSize;
		this.testSize = testSize;

		return this;
	}

	public NeuralNetBuilder setModelStorage(String storageLocation, ModelStorage modelStorage) {
		this.modelStorage = modelStorage;
		this.storageLocation = storageLocation;

		return this;
	}

	public NeuralNetBuilder loadModel(String location, ModelStorage modelStorage) throws Exception {
		this.modelStorage = modelStorage;
		this.neuralNet = this.modelStorage.loadModel(location);
		this.correction = this.neuralNet.getCorrection();

		return this;
	}

	public NeuralNetBuilder build() {
		this.neuralNet = new NeuralNet(
				this.inputSize,
				this.outputSize,
				this.hLayers,
				this.hLayersSize,
				this.learningRate,
				this.defaultInitialization,
				this.outputInitialization,
				this.defaultStrategy,
				this.outputStrategy,
				this.correction);

		return this;
	}

	public NeuralNetBuilder train() throws Exception {
		this.neuralNet.trainModel(this.trainingData, this.batchSize);

		return this;
	}

	public NeuralNetBuilder test() throws Exception {
		float error = 0;
		for (int x = 0; x < this.testSize; x++) {
			TrainingData data = this.trainingData.get(x);

			float[] output = this.neuralNet.runModel(data.getInputData());

			error += this.correction.computeError(output, data.getOutputData());
		}

		double score = 100.0 - ((error / this.testSize) * 100);
		System.out.println("Accuracy: " + score + "%");

		return this;
	}

	public NeuralNetBuilder run(List<TrainingData> trainingData) throws Exception {
		for (TrainingData data : trainingData) {
			float[] output = this.neuralNet.runModel(data.getInputData());

			System.out.println("Output: " + Arrays.toString(output));
		}

		return this;
	}

	public NeuralNetBuilder save() throws Exception {
		this.modelStorage.saveModel(this.neuralNet, this.storageLocation);

		return this;
	}

	public NeuralNet getNeuralNet() {
		return neuralNet;
	}
}
