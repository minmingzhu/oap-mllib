package com.intel.oneapi.dal.table;


import org.junit.Test;

import static com.intel.oneapi.dal.table.Common.DataLayout.*;
import static com.intel.oneapi.dal.table.Common.DataType.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;


public class HomogenTableSuite {
    private static final float MAXIMUMFLOATDELTA = 0.000001f;
    private static final double MAXIMUMDOUBLEDELTA = 0.000001d;

    @Test
    public void createEmptyTable() {
        HomogenTable table = new HomogenTable();
        assertEquals(false, table.hasData());
        assertEquals(new Long(0), table.getColumnCount());
        assertEquals(new Long(0), table.getRowCount());
        assertEquals(new Long(1), table.getKind());
    }
    @Test
    // can construct rowmajor int table 5x2
    public void createRowmajorIntTable() throws Exception {
        int[] data = {1, 2, 3, 4, 5, 6, 10, 80, 10, 11};
        HomogenTable table = new HomogenTable(5, 2,
                data, Integer.class, ROWMAJOR.ordinal());

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());
        assertEquals(Common.DataLayout.ROWMAJOR,table.getDataLayout());

        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), INT32);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ORDINAL);
        }

        assertArrayEquals(data, table.getIntData());
    }
    @Test
    // can construct rowmajor double table 5x2
    public void createRowmajorDoubleTable() throws Exception {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, Double.class, ROWMAJOR.ordinal());

        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());
        assertEquals(Common.DataLayout.ROWMAJOR,table.getDataLayout());

        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), FLOAT64);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.RATIO);
        }

        assertArrayEquals("", data, table.getDoubleData(), MAXIMUMDOUBLEDELTA);
    }
    @Test
    // can construct rowmajor long table 5x2
    public void createRowmajorLongTable() throws Exception {
        long[] data = {1L, 2L, 3L, 4L, 5L, 6L, 10L, 80L, 10L, 11L};
        HomogenTable table = new HomogenTable(5, 2,
                data, Long.class, ROWMAJOR.ordinal());
        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());
        assertEquals(Common.DataLayout.ROWMAJOR,table.getDataLayout());

        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), INT64);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ORDINAL);
        }

        assertArrayEquals(data, table.getLongData());
    }
    @Test
    // can construct rowmajor float table 5x2
    public void createRowmajorFloatTable() throws Exception {
        float[] data = {5.236359f, 8.718667f, 40.724176f, 10.770023f, 90.119887f, 3.815366f,
                53.620204f, 33.219769f, 85.208661f, 15.966239f};
        HomogenTable table = new HomogenTable(5, 2,
                data, Float.class, ROWMAJOR.ordinal());
        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(Common.DataLayout.ROWMAJOR,table.getDataLayout());

        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), FLOAT32);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.RATIO);
        }

        assertArrayEquals("", data, table.getFloatData(), MAXIMUMFLOATDELTA);
    }

    @Test
    // can construct colmajor int table
    public void createColmajorIntTable() throws Exception {
        int[] data = {1, 2, 3, 4, 5, 6, 10, 80, 10, 11};
        HomogenTable table = new HomogenTable(5, 2,
                data, Integer.class, COLUMNMAJOR.ordinal());
        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(COLUMNMAJOR,table.getDataLayout());
        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), INT32);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ORDINAL);
        }
        assertArrayEquals(data, table.getIntData());
    }

    @Test
    // can construct colmajor float table
    public void createColmajorFloatTable() throws Exception {
        float[] data = {5.236359f, 8.718667f, 40.724176f, 10.770023f, 90.119887f, 3.815366f,
                53.620204f, 33.219769f, 85.208661f, 15.966239f};
        HomogenTable table = new HomogenTable(5, 2,
                data, Float.class, COLUMNMAJOR.ordinal());
        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(COLUMNMAJOR,table.getDataLayout());
        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), FLOAT32);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.RATIO);
        }
        assertArrayEquals("", data, table.getFloatData(), MAXIMUMFLOATDELTA);
    }

    @Test
    // can construct colmajor long table
    public void createColmajorLongTable() throws Exception {
        long[] data = {1L, 2L, 3L, 4L, 5L, 6L, 10L, 80L, 10L, 11L};
        HomogenTable table = new HomogenTable(5, 2,
                data, Long.class, COLUMNMAJOR.ordinal());
        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(COLUMNMAJOR,table.getDataLayout());
        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), INT64);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.ORDINAL);
        }
        assertArrayEquals(data, table.getLongData());
    }

    @Test
    // can construct colmajor double table
    public void createColmajorDoubleTable() throws Exception {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, Double.class, COLUMNMAJOR.ordinal());
        assertEquals(true, table.hasData());
        assertEquals(new Long(2), table.getColumnCount());
        assertEquals(new Long(5), table.getRowCount());

        assertEquals(COLUMNMAJOR,table.getDataLayout());
        TableMetadata metadata = table.getMetaData();
        for (int i =0; i < 2; i++) {
            assertEquals(metadata.getDataType(i), FLOAT64);
            assertEquals(metadata.getFeatureType(i), Common.FeatureType.RATIO);
        }
        assertArrayEquals("", data, table.getDoubleData(), MAXIMUMDOUBLEDELTA);
    }

    @Test
    public void testMergeHomogenTable() throws Exception {
        double[] data1 = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        double[] data2 = {7.279464d,0.390664d,-9.619284d,3.435376d,-4.769490d,-4.873188d,-0.118791d,
                -5.117316d,-0.418655d,-0.475422d};
        HomogenTable table1 = new HomogenTable(5, 2,
                data1, Double.class, COLUMNMAJOR.ordinal());
        HomogenTable table2 = new HomogenTable(5, 2,
                data2, Double.class, COLUMNMAJOR.ordinal());

        table1.addHomogenTable(table2.getcObejct());
        double[] expect = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d,7.279464d,0.390664d,-9.619284d,
                3.435376d,-4.769490d,-4.873188d,-0.118791d,-5.117316d,-0.418655d,-0.475422d};
        double[] result = table1.getDoubleData();

        assertArrayEquals("", result, expect, MAXIMUMDOUBLEDELTA);
    }
}
