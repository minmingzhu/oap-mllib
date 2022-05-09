package com.intel.oneapi.dal.table;

public class CommonTest {
    public static Common.ComputeDevice getDevice(){
        String device = System.getProperty("computeDevice");
        if(device == null){
            device = "CPU";
        }
        System.out.println("getDevice : " + device);
        return Common.ComputeDevice.getOrdinalByName(device);
    }
}
