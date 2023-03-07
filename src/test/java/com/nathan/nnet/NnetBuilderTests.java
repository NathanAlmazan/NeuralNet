package com.nathan.nnet;

import com.nathan.nnet.correction.MeanSquareError;
import com.nathan.nnet.initialization.HeInitial;
import com.nathan.nnet.initialization.NormalizedXavier;
import com.nathan.nnet.store.JsonStorage;
import com.nathan.nnet.strategies.ReLu;
import com.nathan.nnet.strategies.Sigmoid;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class NnetBuilderTests {

	@Test
	void testNeuralNet() throws Exception {
		NeuralNet neuralNet = new NeuralNet(7, 10, 2, 8, 0.1f, new HeInitial(), new NormalizedXavier(), new ReLu(), new Sigmoid(), new MeanSquareError());

		// training data
		List<TrainingData> trainingData = new ArrayList<>();

		float[][] inputs = {
				{ 0, 1, 0, 0, 1, 0, 0 },
				{ 1, 1, 1, 0, 0, 1, 1 },
				{ 1, 1, 1, 0, 1, 1, 0 },
				{ 0, 1, 1, 1, 1, 0, 0 },
				{ 1, 0, 1, 1, 1, 1, 0 },
				{ 1, 0, 1, 1, 1, 1, 1 },
				{ 1, 1, 0, 0, 1, 0, 0 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 0, 0 },
				{ 1, 1, 0, 1, 1, 1, 1 }
		};

		float[][] tests = {
				{ 0, 1, 0, 0, 1, 0, 0 },
				{ 1, 1, 1, 0, 0, 1, 1 },
				{ 1, 1, 1, 0, 1, 1, 0 },
				{ 0, 1, 1, 1, 1, 0, 0 },
				{ 1, 0, 1, 1, 1, 1, 0 },
				{ 1, 0, 1, 1, 1, 1, 1 },
				{ 1, 1, 0, 0, 1, 0, 0 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 0, 0 },
				{ 1, 1, 0, 1, 1, 1, 1 }
		};

		float[][] outputs = {
				{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
				{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
		};

		for (int t = 0; t < 10000; t++) {
			for (int x = 0; x < inputs.length; x++) trainingData.add(new TrainingData(inputs[x], outputs[x]));
		}

		System.out.println(trainingData.size());

		neuralNet.trainModel(trainingData, 48);

		System.out.println("Tests: ");

		for (float[] data : tests) {
			float[] output = neuralNet.runModel(data);

			System.out.println("Answer: " + Arrays.toString(output));
		}

		System.out.println("Weights and Biases: ");
		neuralNet.printWeightsAndBiases();
	}

	@Test
	void buildNNet() throws Exception {
		// training data
		List<TrainingData> trainingData = new ArrayList<>();

		float[][] inputs = {
				{ 0, 1, 0, 0, 1, 0, 0 },
				{ 1, 1, 1, 0, 0, 1, 1 },
				{ 1, 1, 1, 0, 1, 1, 0 },
				{ 0, 1, 1, 1, 1, 0, 0 },
				{ 1, 0, 1, 1, 1, 1, 0 },
				{ 1, 0, 1, 1, 1, 1, 1 },
				{ 1, 1, 0, 0, 1, 0, 0 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 0, 0 },
				{ 1, 1, 0, 1, 1, 1, 1 }
		};

		float[][] outputs = {
				{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
				{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
		};

		for (int t = 0; t < 5000; t++) {
			for (int x = 0; x < inputs.length; x++) trainingData.add(new TrainingData(inputs[x], outputs[x]));
		}

		NeuralNetBuilder neuralNetBuilder = new NeuralNetBuilder();
		neuralNetBuilder
			.setDimensions(7, 10)
			.setHiddenLayers(2, 8)
			.setWeightInitialization(new HeInitial(), new NormalizedXavier())
			.setLearningStrategy(new ReLu(), new Sigmoid())
			.setErrorCorrection(0.1f, new MeanSquareError())
			.build()
			.setTrainingParameters(trainingData, 50,100)
			.train()
			.train()
			.test()
			.setModelStorage("E:\\ML\\tests\\digits.json", new JsonStorage())
			.save();
	}

	@Test
	void loadAndTestModel() throws Exception {
		// training data
		List<TrainingData> trainingData = new ArrayList<>();

		float[][] inputs = {
				{ 0, 1, 0, 0, 1, 0, 0 },
				{ 1, 1, 1, 0, 0, 1, 1 },
				{ 1, 1, 1, 0, 1, 1, 0 },
				{ 0, 1, 1, 1, 1, 0, 0 },
				{ 1, 0, 1, 1, 1, 1, 0 },
				{ 1, 0, 1, 1, 1, 1, 1 },
				{ 1, 1, 0, 0, 1, 0, 0 },
				{ 1, 1, 1, 1, 1, 1, 1 },
				{ 1, 1, 1, 1, 1, 0, 0 },
				{ 1, 1, 0, 1, 1, 1, 1 }
		};

		float[][] outputs = {
				{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
				{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
				{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
		};

		for (int x = 0; x < inputs.length; x++) trainingData.add(new TrainingData(inputs[x], outputs[x]));

		NeuralNetBuilder neuralNetBuilder = new NeuralNetBuilder();
		neuralNetBuilder
				.loadModel("E:\\ML\\tests\\digits.json", new JsonStorage())
				.setTrainingParameters(trainingData, 50,10)
				.test();
	}
}
