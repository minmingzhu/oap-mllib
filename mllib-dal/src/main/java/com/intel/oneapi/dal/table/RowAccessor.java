package com.intel.oneapi.dal.table;

public class RowAccessor {
    private long cObject;

    public RowAccessor(long cObject) {
        this.cObject = cObject;
    }

    public double[] pullDouble(){
        return this.cPullDouble(this.cObject, 0, -1);
    }

    public double[] pullDouble(long rowStartIndex, long rowEndIndex){
        return this.cPullDouble(this.cObject, rowStartIndex, rowEndIndex);
    }

    private native double[] cPullDouble(long cObject, long cRowStartIndex, long cRowEndIndex);

    public float[] pullFloat(){
        return this.cPullFloat(this.cObject, 0, -1);
    }

    public float[] pullFloat(long rowStartIndex, long rowEndIndex){
        return this.cPullFloat(this.cObject, rowStartIndex, rowEndIndex);
    }

    private native float[] cPullFloat(long cObject, long cRowStartIndex, long cRowEndIndex);

    public int[] pullInt(){
        return this.cPullInt(this.cObject, 0, -1);
    }

    public int[] pullInt(long rowStartIndex, long rowEndIndex){
        return this.cPullInt(this.cObject, rowStartIndex, rowEndIndex);
    }

    private native int[] cPullInt(long cObject, long cRowStartIndex, long cRowEndIndex);

    public long[] pullLong(){
        return this.cPullLong(this.cObject, 0, -1);
    }

    public long[] pullLong(long rowStartIndex, long rowEndIndex){
        return this.cPullLong(this.cObject, rowStartIndex, rowEndIndex);
    }

    private native long[] cPullLong(long cObject, long cRowStartIndex, long cRowEndIndex);
}
