package com.nathan.nnet;

import com.nathan.nnet.correction.ErrorCorrection;
import com.nathan.nnet.dataset.DatasetLoader;
import com.nathan.nnet.dataset.NormalizeData;
import com.nathan.nnet.store.ModelStorage;

import java.util.Arrays;
import java.util.List;

public class NeuralNetBuilder {
	private float learningRate;
	private List<NLayer> layers;
	private ErrorCorrection correction;
	private NeuralNet neuralNet;
	private List<TrainingData> trainingData;
	private int batchSize;
	private int testSize;
	private String storageLocation;
	private ModelStorage modelStorage;

	public void setLayers(List<NLayer> layers) {
		this.layers = layers;
	}

	public NeuralNetBuilder setErrorCorrection(float learningRate, ErrorCorrection correction) {
		this.learningRate = learningRate;
		this.correction = correction;

		return this;
	}

	public NeuralNetBuilder setTrainingParameters(
			List<TrainingData> trainingData,
			int batchSize,
			int testSize
	) {
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
		this.neuralNet = this.modelStorage.loadModel(location, null);
		this.correction = this.neuralNet.getCorrection();

		return this;
	}

	public NeuralNetBuilder loadModel(String location, ModelStorage modelStorage, float learningRate) throws Exception {
		this.modelStorage = modelStorage;
		this.neuralNet = this.modelStorage.loadModel(location, learningRate);
		this.correction = this.neuralNet.getCorrection();

		return this;
	}

	public NeuralNetBuilder build() {
		this.neuralNet = new NeuralNet(
				this.learningRate,
				this.layers,
				this.correction
		);

		return this;
	}

	public NeuralNetBuilder train() throws Exception {
		this.neuralNet.trainModel(this.trainingData.subList(this.testSize, this.trainingData.size()), this.batchSize);

		return this;
	}

	public NeuralNetBuilder train(int repeat) throws Exception {
		for (int x = 0; x < repeat; x++)
			this.neuralNet.trainModel(this.trainingData.subList(this.testSize, this.trainingData.size()), this.batchSize);

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
			System.out.println("Expected Output: " + Arrays.toString(data.getOutputData()));
			System.out.println();
		}

		return this;
	}

	public NeuralNetBuilder save() throws Exception {
		this.modelStorage.saveModel(this.neuralNet, this.storageLocation);

		return this;
	}
}
