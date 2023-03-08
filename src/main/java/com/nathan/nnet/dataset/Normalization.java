package com.nathan.nnet.dataset;

import java.util.ArrayList;
import java.util.List;

public class Normalization implements NormalizeData {
    @Override
    public List<Float> normalizeColumn(List<Float> vector) {
        int upper = 1;
        int lower = 0;

        // find max
        float max = 0;
        for (Float num : vector) {
            if (num > max) max = num;
        }

        // find min
        float min = max;
        for (Float num : vector) {
            if (num < min) min = num;
        }

        List<Float> newVector = new ArrayList<>();

        for (Float num : vector) {
            float norm = ((num - min) / (max - min)) * (upper - lower) + lower;
            newVector.add(norm);
        }

        return newVector;
    }
}
