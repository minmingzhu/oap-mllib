package com.intel.oneapi.dal.table;

public class RowAccessor {
    private transient long cObject;

    public RowAccessor(long cObject) {
        this.cObject = cObject;
    }

    public double[] pullDouble(Common.ComputeDevice device){
        return this.cPullDouble(this.cObject, 0, -1, device.ordinal());
    }

    public double[] pullDouble(long rowStartIndex, long rowEndIndex, Common.ComputeDevice device){
        return this.cPullDouble(this.cObject, rowStartIndex, rowEndIndex, device.ordinal());
    }

    private native double[] cPullDouble(long cObject, long cRowStartIndex,
                                        long cRowEndIndex, int computeDeviceIndex);

    public float[] pullFloat(Common.ComputeDevice device){
        return this.cPullFloat(this.cObject, 0, -1, device.ordinal());
    }

    public float[] pullFloat(long rowStartIndex, long rowEndIndex, Common.ComputeDevice device){
        return this.cPullFloat(this.cObject, rowStartIndex, rowEndIndex, device.ordinal());
    }

    private native float[] cPullFloat(long cObject, long cRowStartIndex,
                                      long cRowEndIndex, int computeDeviceIndex);

    public int[] pullInt(Common.ComputeDevice device){
        return this.cPullInt(this.cObject, 0, -1, device.ordinal());
    }

    public int[] pullInt(long rowStartIndex, long rowEndIndex, Common.ComputeDevice device){
        return this.cPullInt(this.cObject, rowStartIndex, rowEndIndex, device.ordinal());
    }

    private native int[] cPullInt(long cObject, long cRowStartIndex,
                                  long cRowEndIndex, int computeDeviceIndex);

    public long[] pullLong(Common.ComputeDevice device){
        return this.cPullLong(this.cObject, 0, -1, device.ordinal());
    }

    public long[] pullLong(long rowStartIndex, long rowEndIndex, Common.ComputeDevice device){
        return this.cPullLong(this.cObject, rowStartIndex, rowEndIndex, device.ordinal());
    }

    private native long[] cPullLong(long cObject, long cRowStartIndex,
                                    long cRowEndIndex, int computeDeviceIndex);

}
