package com.intel.oap.mllib;

import au.com.bytecode.opencsv.CSVReader;
import com.intel.oap.mllib.clustering.KMeansDALImpl;
import com.intel.oap.mllib.clustering.KMeansResult;
import com.intel.oneapi.dal.table.Common;
import com.intel.oneapi.dal.table.HomogenTable;
import org.junit.jupiter.api.Test;

import java.io.FileReader;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class KmeansHomegenTableTest {
    @Test
    public void computeKmenasHomogenTable() throws Exception {
        // Create an object of filereader
        // class with CSV file as a parameter.
        
        FileReader datareader = new FileReader("src/test/java/com/intel/oap/mllib/data/kmeans_dense_train_centroids.csv");
        FileReader centroidsreader = new FileReader("src/test/java/com/intel/oap/mllib/data/kmeans_dense_train_centroids.csv");

        // create csvReader object passing
        // file reader as a parameter
        CSVReader dataCsvReader = new CSVReader(datareader);
        CSVReader centroidsCsvReader = new CSVReader(centroidsreader);
        List<String[]> dataAllRows = dataCsvReader.readAll();
        List<String[]> centroidsAllRows = centroidsCsvReader.readAll();

        String[][] dataMatrix = new String[dataAllRows.size()][];
        String[][] centroidsMatrix = new String[centroidsAllRows.size()][];

        for (int i = 0; i < dataAllRows.size(); i++) {
            dataMatrix[i] = dataAllRows.get(i);
        }

        for (int i = 0; i < centroidsAllRows.size(); i++) {
            centroidsMatrix[i] = centroidsAllRows.get(i);
        }

        double[] dataArray = new double[dataMatrix.length * dataMatrix[0].length];
        double[] centroidsArray = new double[centroidsMatrix.length * centroidsMatrix[0].length];
        int index =0 ;
        for (int i = 0; i < dataAllRows.size(); i++) {
            for (int j = 0; j < dataAllRows.get(i).length; j++) {
                dataArray[index] = Double.parseDouble(dataAllRows.get(i)[j]);
                index++;
            }
        }
        index =0 ;
        for (int i = 0; i < centroidsAllRows.size(); i++) {
            for (int j = 0; j < centroidsAllRows.get(i).length; j++) {
                centroidsArray[index] = Double.parseDouble(centroidsAllRows.get(i)[j]);
                index++;
            }
        }

        HomogenTable dataTable = new HomogenTable(dataMatrix.length, dataMatrix[0].length, dataArray, Common.ComputeDevice.HOST);
        HomogenTable centroidsTable = new HomogenTable(centroidsMatrix.length, centroidsMatrix[0].length, centroidsArray, Common.ComputeDevice.HOST);

        KMeansDALImpl kmeansDAL = new KMeansDALImpl(0, 0, 0,
               null, null, 0, 0);
        KMeansResult result = new KMeansResult();
        kmeansDAL.cKMeansOneapiComputeWithInitCenters(dataTable.getcObejct(), centroidsTable.getcObejct(),20, 0.001,
                5, 1, Common.ComputeDevice.HOST.ordinal(), 0, "127.0.0.1_3000" , result);
    }
}
