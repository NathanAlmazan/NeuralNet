package com.nathan.nnet.dataset;

import com.nathan.nnet.TrainingData;

import java.util.List;

public interface DatasetLoader {
    Dataset loadDataset(String location, String targetColumn, NormalizeData normalize) throws Exception;
}
