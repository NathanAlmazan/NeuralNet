package com.nathan.nnet.store;

import com.nathan.nnet.correction.CrossEntropy;
import com.nathan.nnet.correction.ErrorCorrection;
import com.nathan.nnet.correction.MeanSquareError;
import com.nathan.nnet.initialization.*;
import com.nathan.nnet.strategies.*;

import java.util.HashMap;
import java.util.Map;

public abstract class StringToObject {
    public static HashMap<String, Strategy> STRATEGY = new HashMap<>(Map.of(
            "LeakyReLu", new LeakyReLu(),
            "ReLu", new ReLu(),
            "Sigmoid", new Sigmoid(),
            "Softmax", new Softmax(),
            "Tanh", new Tanh()
    ));

    public static HashMap<String, ErrorCorrection> CORRECTION = new HashMap<>(Map.of(
            "MeanSquareError", new MeanSquareError(),
            "CrossEntropy", new CrossEntropy()
    ));

    public static HashMap<String, WeightInitialization> INITIALIZATION = new HashMap<>(Map.of(
            "HeInitial", new HeInitial(),
            "Xavier", new Xavier(),
            "NormalizedXavier", new NormalizedXavier(),
            "NormalRandom", new NormalRandom()
    ));
}
