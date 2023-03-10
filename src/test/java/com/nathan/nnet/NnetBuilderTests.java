package com.nathan.nnet;

import com.nathan.nnet.correction.MeanSquareError;
import com.nathan.nnet.dataset.CSV;
import com.nathan.nnet.dataset.Normalization;
import com.nathan.nnet.initialization.HeInitial;
import com.nathan.nnet.initialization.NormalizedXavier;
import com.nathan.nnet.store.JsonStorage;
import com.nathan.nnet.strategies.LeakyReLu;
import com.nathan.nnet.strategies.ReLu;
import com.nathan.nnet.strategies.Sigmoid;
import org.junit.jupiter.api.Test;

import java.util.List;

class NnetBuilderTests {

	@Test
	void loadAndTestModel() throws Exception {
		CSV csv = new CSV();
		List<TrainingData> dataset = csv.loadDataset("E:\\ML\\datasets\\diabetes2.csv", "Outcome", new Normalization());

		NeuralNetBuilder neuralNetBuilder = new NeuralNetBuilder();
		neuralNetBuilder
				.loadModel("E:\\ML\\tests\\diabetes.json", new JsonStorage())
				.run(dataset.subList(200, 220));
	}

	@Test
	void loadAndTrainModel() throws Exception {
		CSV csv = new CSV();
		List<TrainingData> dataset = csv.loadDataset("E:\\ML\\datasets\\diabetes2.csv", "Outcome", new Normalization());

		NeuralNetBuilder neuralNetBuilder = new NeuralNetBuilder();
		neuralNetBuilder
				.loadModel("E:\\ML\\tests\\diabetes.json", new JsonStorage(), 0.005f)
				.setTrainingParameters(dataset, 167,100)
				.train(100000)
				.test()
				.setModelStorage("E:\\ML\\tests\\diabetes.json", new JsonStorage())
				.save();
	}

	@Test
	void loadAndTrainCsv() throws Exception {
		CSV csv = new CSV();
		List<TrainingData> dataset = csv.loadDataset("E:\\ML\\datasets\\diabetes2.csv", "Outcome", new Normalization());

		NeuralNetBuilder neuralNetBuilder = new NeuralNetBuilder();
		neuralNetBuilder
				.setDimensions(8, 1)
				.setHiddenLayers(2, 3)
				.setWeightInitialization(new NormalizedXavier(), new NormalizedXavier())
				.setLearningStrategy(new Sigmoid(), new Sigmoid())
				.setErrorCorrection(0.01f, new MeanSquareError())
				.build()
				.setTrainingParameters(dataset, 167, 100)
				.train(100000)
				.test()
				.setModelStorage("E:\\ML\\tests\\diabetes.json", new JsonStorage())
				.save();
	}
}
