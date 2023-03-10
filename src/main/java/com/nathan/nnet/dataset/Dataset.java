package com.nathan.nnet.dataset;

import java.util.List;

public class Dataset {
    private final List<String> columns;
    private final List<List<String>> rows;

    public Dataset(List<String> columns, List<List<String>> rows) {
        this.columns = columns;
        this.rows = rows;
    }

    public List<String> getColumns() {
        return columns;
    }

    public List<List<String>> getRows() {
        return rows;
    }
}
