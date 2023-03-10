package com.nathan.nnet.dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class CSV implements DatasetLoader {

    private static final String COMMA_DELIMITER = ",";

    @Override
    public Dataset loadDataset(String location, String targetColumn, NormalizeData normalize) throws Exception {
        List<List<String>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(location))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(COMMA_DELIMITER);
                records.add(Arrays.asList(values));
            }
        }

        List<String> header = records.remove(0);

        return new Dataset(header, records);
    }
}
