package com.intel.oneapi.dal.table;

public class ColumnAccessor {
    private long cObject;

    public ColumnAccessor(long cObject) {
        this.cObject = cObject;
    }

    public double[] pullDouble(long columnIndex){
        return this.cPullDouble(this.cObject, columnIndex, 0, -1);
    }

    public double[] pullDouble(long columnIndex, long rowStartIndex, long rowEndIndex){
        return this.cPullDouble(this.cObject, columnIndex, rowStartIndex, rowEndIndex);
    }

    private native double[] cPullDouble(long cObject, long cColumnIndex, long cRowStartIndex, long cRowEndIndex);

    public float[] pullFloat(long columnIndex){
        return this.cPullFloat(this.cObject, columnIndex, 0, -1);
    }

    public float[] pullFloat(long columnIndex, long rowStartIndex, long rowEndIndex){
        return this.cPullFloat(this.cObject, columnIndex, rowStartIndex, rowEndIndex);
    }

    private native float[] cPullFloat(long cObject, long cColumnIndex, long cRowStartIndex, long cRowEndIndex);

    public int[] pullInt(long columnIndex){
        return this.cPullInt(this.cObject, columnIndex, 0, -1);
    }

    public int[] pullInt(long columnIndex, long rowStartIndex, long rowEndIndex){
        return this.cPullInt(this.cObject, columnIndex, rowStartIndex, rowEndIndex);
    }

    private native int[] cPullInt(long cObject, long cColumnIndex, long cRowStartIndex, long cRowEndIndex);

    public long[] pullLong(long columnIndex){
        return this.cPullLong(this.cObject, columnIndex, 0, -1);
    }

    public long[] pullLong(long columnIndex, long rowStartIndex, long rowEndIndex){
        return this.cPullLong(this.cObject, columnIndex, rowStartIndex, rowEndIndex);
    }

    private native long[] cPullLong(long cObject, long cColumnIndex, long cRowStartIndex, long cRowEndIndex);
}
