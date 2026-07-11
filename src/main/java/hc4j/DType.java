package hc4j;

import java.lang.foreign.ValueLayout;

public enum DType{
    i32(ValueLayout.JAVA_INT),
    f32(ValueLayout.JAVA_FLOAT),
    f64(ValueLayout.JAVA_DOUBLE);

    public final ValueLayout layout;
    DType(ValueLayout layout){
        this.layout = layout;
    }
}