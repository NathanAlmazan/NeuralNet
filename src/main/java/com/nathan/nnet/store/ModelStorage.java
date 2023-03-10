package com.nathan.nnet.store;

import com.nathan.nnet.NeuralNet;


public interface ModelStorage {

    NeuralNet loadModel(String location, Float rate) throws Exception;

    void saveModel(NeuralNet network, String location) throws Exception;

}
