package com.intel.oneapi.dal.table;

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import org.junit.jupiter.api.Test;
import java.lang.reflect.Array;
import java.util.Arrays;

import static com.intel.oneapi.dal.table.Common.DataLayout.ROW_MAJOR;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class RowAccessorSuite {
=======
=======
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
>>>>>>> 1. convert homogenTable to array/vector/matrix
import org.junit.Test;
=======
import org.junit.jupiter.api.Test;
>>>>>>> rollback to edb9f3d

import java.lang.reflect.Array;
import java.util.Arrays;

import static com.intel.oneapi.dal.table.Common.DataLayout.ROW_MAJOR;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class RowAccessorSuite {
<<<<<<< HEAD
    private static final double MAXIMUMDOUBLEDELTA = 0.000001d;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
>>>>>>> rollback to edb9f3d

    @Test
    public void readTableDataFromRowAccessor() {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
                data, ROW_MAJOR, CommonTest.getDevice());

        RowAccessor accessor = new RowAccessor(table.getcObejct());
        double[] rowData = accessor.pullDouble( 0 , table.getRowCount(), CommonTest.getDevice());
        assertEquals(new Long(rowData.length), new Long(table.getColumnCount() * table.getRowCount()));
        assertArrayEquals(rowData, data);
        for (int i = 0; i < rowData.length; i++) {
            assertEquals(rowData[i], data[i]);
=======
                data, Double.class, ROWMAJOR.ordinal());
=======
                data, ROWMAJOR.ordinal());
>>>>>>> Merge branch 'make_homogen_table' into convert_homogentable
=======
                data, Double.class, ROWMAJOR.ordinal());
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
                data, ROWMAJOR.ordinal());
>>>>>>> Merge branch 'make_homogen_table' into convert_homogentable
=======
                data, Double.class, ROWMAJOR.ordinal());
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
                data, ROWMAJOR.ordinal());
>>>>>>> Merge branch 'make_homogen_table' into convert_homogentable
=======
                data, Double.class, ROWMAJOR.ordinal());
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
                data, ROW_MAJOR, CommonTest.getDevice());
>>>>>>> rollback to edb9f3d

        RowAccessor accessor = new RowAccessor(table.getcObejct());
        double[] rowData = accessor.pullDouble( 0 , table.getRowCount());
        assertEquals(new Long(rowData.length), new Long(table.getColumnCount() * table.getRowCount()));
        assertArrayEquals(rowData, data);
        for (int i = 0; i < rowData.length; i++) {
<<<<<<< HEAD
            assertEquals(rowData[i], data[i], MAXIMUMDOUBLEDELTA);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
>>>>>>> 1. convert homogenTable to array/vector/matrix
=======
            assertEquals(rowData[i], data[i]);
>>>>>>> rollback to edb9f3d
        }
    }
}
