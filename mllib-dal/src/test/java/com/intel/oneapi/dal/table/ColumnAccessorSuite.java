package com.intel.oneapi.dal.table;

import org.junit.Test;

import static com.intel.oneapi.dal.table.Common.DataLayout.ROWMAJOR;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class ColumnAccessorSuite {
    private static final double MAXIMUMDOUBLEDELTA = 0.000001d;

    @Test
    public void getFirstColumnFromHomogenTable() throws Exception {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, ROWMAJOR.ordinal());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct());
        double[] columnData = accessor.pullDouble(0);
        assertEquals(new Long(columnData.length), table.getRowCount());
        double[] tableData = table.getDoubleData();
        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[i * table.getColumnCount().intValue()], MAXIMUMDOUBLEDELTA);
        }
    }

    @Test
    public void getSecondColumnFromHomogenTableWithConversion() throws Exception {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, ROWMAJOR.ordinal());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct());
        double[] columnData = accessor.pullDouble(1);
        assertEquals(new Long(columnData.length), table.getRowCount());
        double[] tableData = table.getDoubleData();

        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[i * table.getColumnCount().intValue() + 1], MAXIMUMDOUBLEDELTA);
        }
    }

    @Test
    public void getSecondColumnFromHomogenTableWithSubsetOfRows() throws Exception {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, ROWMAJOR.ordinal());

        ColumnAccessor accessor = new ColumnAccessor(table.getcObejct());
        double[] columnData = accessor.pullDouble(0, 1 , 3);
        assertEquals(new Long(columnData.length), new Long(2));
        double[] tableData = table.getDoubleData();
        for (int i = 0; i < columnData.length; i++) {
            assertEquals(columnData[i], tableData[2 + i * table.getColumnCount().intValue()], MAXIMUMDOUBLEDELTA);
        }
    }

}
