package com.intel.oneapi.dal.table;

import org.junit.Test;

import java.lang.reflect.Array;
import java.util.Arrays;

import static com.intel.oneapi.dal.table.Common.DataLayout.ROWMAJOR;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class RowAccessorSuite {
    private static final double MAXIMUMDOUBLEDELTA = 0.000001d;

    @Test
    public void readTableDataFromRowAccessor() {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, Double.class, ROWMAJOR.ordinal());

        RowAccessor accessor = new RowAccessor(table.getcObejct());
        double[] rowData = accessor.pullDouble( 0 , table.getRowCount(), CommonTest.getDevice());
        assertEquals(new Long(rowData.length), new Long(table.getColumnCount() * table.getRowCount()));
        assertArrayEquals("", rowData, data, MAXIMUMDOUBLEDELTA);
        for (int i = 0; i < rowData.length; i++) {
            assertEquals(rowData[i], data[i], MAXIMUMDOUBLEDELTA);
        }
    }
}
