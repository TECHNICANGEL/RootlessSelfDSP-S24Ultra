# FLAGSHIP EDITION - Aggressive optimizations
# No compromises, maximum performance

# Aggressive optimization flags
-optimizationpasses 10
-allowaccessmodification
-mergeinterfacesaggressively

# Enable all optimizations
-optimizations !code/simplification/arithmetic,!code/simplification/cast,!field/*,!class/merging/*,!code/allocation/variable

# Flagship-only: Remove all debugging overhead
-assumenosideeffects class android.util.Log {
    public static *** d(...);
    public static *** v(...);
    public static *** i(...);
}

# Keep only essential code
-dontobfuscate

-keep class dev.doubledot.doki.** { *; }

-keep class me.timschneeberger.hiddenapi_impl.** { *; }

# Hidden API refined classes - prevent R8 class descriptor conflicts
-keep class android.media.audiofx.** { *; }
-dontwarn android.media.audiofx.**
-keep class android.media.projection.** { *; }
-dontwarn android.media.projection.**

# Rikka Refine - ignore class descriptor mismatches
-ignorewarnings
-dontwarn dev.rikka.tools.refine.**

-keep,allowoptimization class me.timschneeberger.rootlessjamesdsp.interop.** { *; }
-keep,allowoptimization class me.timschneeberger.rootlessjamesdsp.fragment.** { *; }

-keepclasseswithmembernames class * {
    native <methods>;
}

-keepclasseswithmembernames class * {
    public <init>(android.content.Context, android.util.AttributeSet);
}

-keepclasseswithmembernames class * {
    public <init>(android.content.Context, android.util.AttributeSet, int);
}

# ONNX Runtime - prevent R8 from breaking the native bindings
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
