package com.intel.oneapi.dal.table;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;


public class RowAccessorTest {
    private String getDevice(){
        String device = System.getProperty("computeDevice");
        if(device == null){
            device = "CPU";
        }
        System.out.println("getDevice : " + device);
        return device;
    }

    @Test
    public void readTableDataFromRowAccessor() {
        double[] data = {5.236359d, 8.718667d, 40.724176d, 10.770023d, 90.119887d, 3.815366d,
                53.620204d, 33.219769d, 85.208661d, 15.966239d};
        HomogenTable table = new HomogenTable(5, 2,
                data, CommonTest.getDevice());

        RowAccessor accessor = new RowAccessor(table.getcObejct());
        double[] rowData = accessor.pullDouble( 0 , table.getRowCount(),
                CommonTest.getDevice());
        assertEquals(new Long(rowData.length), new Long(table.getColumnCount() * table.getRowCount()));
        assertArrayEquals(rowData, data);
        for (int i = 0; i < rowData.length; i++) {
            assertEquals(rowData[i], data[i]);
        }
    }
}
