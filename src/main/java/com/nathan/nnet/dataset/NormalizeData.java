package com.nathan.nnet.dataset;


import java.util.List;

public interface NormalizeData {
    List<Float> normalizeColumn(List<Float> vector);
}
