# Hidden API Refined - ProGuard Rules
# These rules are consumed by the app module

-dontobfuscate

# Keep all refined hidden API classes
-keep class android.media.audiofx.** { *; }
-keep class android.media.projection.** { *; }

# Suppress warnings for hidden API classes
-dontwarn android.media.audiofx.**
-dontwarn android.media.projection.**

# Rikka Refine runtime
-keep class dev.rikka.tools.refine.** { *; }
-dontwarn dev.rikka.tools.refine.**