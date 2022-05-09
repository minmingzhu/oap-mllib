package com.intel.oneapi.dal.table;

public class ColumnAccessor {
    private long cObject;

    public ColumnAccessor(long cObject) {
        this.cObject = cObject;
    }

    public double[] pullDouble(long columnIndex, Common.ComputeDevice device){
        return this.cPullDouble(this.cObject, columnIndex, 0, -1, device.ordinal());
    }

    public double[] pullDouble(long columnIndex, long rowStartIndex, long rowEndIndex, Common.ComputeDevice device){
        return this.cPullDouble(this.cObject, columnIndex, rowStartIndex, rowEndIndex, device.ordinal());
    }

    private native double[] cPullDouble(long cObject, long cColumnIndex,
                                        long cRowStartIndex, long cRowEndIndex, int computeDeviceIndex);

    public float[] pullFloat(long columnIndex, int computeDeviceIndex){
        return this.cPullFloat(this.cObject, columnIndex, 0, -1, computeDeviceIndex);
    }

    public float[] pullFloat(long columnIndex, long rowStartIndex, long rowEndIndex, Common.ComputeDevice device){
        return this.cPullFloat(this.cObject, columnIndex, rowStartIndex, rowEndIndex, device.ordinal());
    }

    private native float[] cPullFloat(long cObject, long cColumnIndex, long cRowStartIndex,
                                      long cRowEndIndex, int computeDeviceIndex);

    public int[] pullInt(long columnIndex, Common.ComputeDevice device){
        return this.cPullInt(this.cObject, columnIndex, 0, -1, device.ordinal());
    }

    public int[] pullInt(long columnIndex, long rowStartIndex, long rowEndIndex, Common.ComputeDevice device){
        return this.cPullInt(this.cObject, columnIndex, rowStartIndex, rowEndIndex, device.ordinal());
    }

    private native int[] cPullInt(long cObject, long cColumnIndex, long cRowStartIndex,
                                  long cRowEndIndex, int computeDeviceIndex);

    public long[] pullLong(long columnIndex, Common.ComputeDevice device){
        return this.cPullLong(this.cObject, columnIndex, 0, -1, device.ordinal());
    }

    public long[] pullLong(long columnIndex, long rowStartIndex, long rowEndIndex, Common.ComputeDevice device){
        return this.cPullLong(this.cObject, columnIndex, rowStartIndex, rowEndIndex, device.ordinal());
    }

    private native long[] cPullLong(long cObject, long cColumnIndex, long cRowStartIndex,
                                    long cRowEndIndex, int computeDeviceIndex);
}
