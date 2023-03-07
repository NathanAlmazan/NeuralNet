package com.nathan.nnet.store;

import com.nathan.nnet.NLayer;
import com.nathan.nnet.NeuralNet;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.*;

public class JsonStorage implements ModelStorage {
    @Override
    public NeuralNet loadModel(String location) throws Exception {
        JSONParser parser = new JSONParser();

        FileReader reader = new FileReader(location);
        JSONObject model = (JSONObject)  parser.parse(reader);

        Double learningRate = (Double) model.get("learningRate");
        String correction = (String) model.get("correction");

        JSONArray listOfLayers = (JSONArray) model.get("layers");

        List<NLayer> modelLayers = new ArrayList<>();

        for (Object layerObj : listOfLayers) {
            JSONObject layer = (JSONObject) layerObj;

            Long inputSize = (Long) layer.get("inputSize");
            Long outputSize = (Long) layer.get("outputSize");
            String initialization = (String) layer.get("initialization");
            String strategy = (String) layer.get("strategy");

            JSONArray biases = (JSONArray) layer.get("biases");
            float[] layerBiases = new float[outputSize.intValue()];

            for (int i = 0; i < outputSize.intValue(); i++)
                layerBiases[i] = ((Double) biases.get(i)).floatValue();

            JSONArray neurons = (JSONArray) layer.get("weights");
            float[][] layerWeights = new float[outputSize.intValue()][inputSize.intValue()];

            for (int i = 0; i < outputSize.intValue(); i++) {
                JSONArray weights = (JSONArray) neurons.get(i);
                float[] layerWeight = new float[inputSize.intValue()];

                for (int j = 0; j < inputSize.intValue(); j++)
                    layerWeight[j] = ((Double) weights.get(j)).floatValue();

                layerWeights[i] = layerWeight;
            }

            modelLayers.add(new NLayer(
                    inputSize.intValue(),
                    outputSize.intValue(),
                    layerWeights,
                    layerBiases,
                    learningRate.floatValue(),
                    StringToObject.INITIALIZATION.get(initialization),
                    StringToObject.STRATEGY.get(strategy)
            ));
        }

        return new NeuralNet(
                learningRate.floatValue(),
                modelLayers,
                StringToObject.CORRECTION.get(correction)
        );
    }

    @Override
    public void saveModel(NeuralNet network, String location) throws Exception {
        JSONObject model = new JSONObject();

        model.put("learningRate", network.getLearningRate());
        model.put("correction", network.getCorrection().getClass().getSimpleName());

        JSONArray listOfLayers = new JSONArray();

        for (NLayer layer : network.getLayers()) {
            JSONObject obj = new JSONObject();

            obj.put("inputSize", layer.getInputSize());
            obj.put("outputSize", layer.getOutputSize());
            obj.put("initialization", layer.getInitialization().getClass().getSimpleName());
            obj.put("strategy", layer.getStrategy().getClass().getSimpleName());

            JSONArray biases = new JSONArray();
            biases.addAll(floatArrayToList(layer.getBiases()));

            obj.put("biases", biases);

            JSONArray weights = new JSONArray();
            for (float[] weight : layer.getWeights()) {
                JSONArray neuron = new JSONArray();
                neuron.addAll(floatArrayToList(weight));

                weights.add(neuron);
            }

            obj.put("weights", weights);

            listOfLayers.add(obj);
        }

        model.put("layers", listOfLayers);


        FileWriter file = new FileWriter(location);
        file.write(model.toJSONString());
        file.flush();
    }

    private List<Float> floatArrayToList(float[] array) {
        List<Float> list = new ArrayList<>();

        for (float num : array) list.add(num);

        return list;
    }
}
