package com.nathan.nnet.dataset;

import com.nathan.nnet.TrainingData;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class CSV implements DatasetLoader {

    private static final String COMMA_DELIMITER = ",";

    @Override
    public List<TrainingData> loadDataset(String location, String targetColumn, NormalizeData normalize) throws Exception {
        HashMap<String, List<Float>> dataset = new HashMap<>();

        List<List<String>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(location))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(COMMA_DELIMITER);
                records.add(Arrays.asList(values));
            }
        }

        List<String> header = records.remove(0);

        // normalized values and collect classes
        for (int x = 0; x < header.size(); x++) {
            String head = header.get(x);

            List<Float> values = new ArrayList<>();

            for (List<String> row : records)
                values.add(Float.parseFloat(row.get(x)));

            dataset.put(head, normalize.normalizeColumn(values));
        }

        List<TrainingData> trainingData = new ArrayList<>();

        for (int x = 0; x < records.size(); x++) {
            float[] input = new float[dataset.size() - 1];
            float[] output = new float[1];

            int index = 0;
            for (String head : header) {
                if (head.equals(targetColumn)) output[0] = dataset.get(head).get(x);
                else input[index] = dataset.get(head).get(x);

                index++;
            }

            trainingData.add(new TrainingData(input, output));
        }

        return trainingData;
    }

}
