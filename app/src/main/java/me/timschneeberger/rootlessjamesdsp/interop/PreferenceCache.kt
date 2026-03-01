package me.timschneeberger.rootlessjamesdsp.interop

import android.content.Context
import android.content.SharedPreferences
import androidx.annotation.StringRes
import me.timschneeberger.rootlessjamesdsp.flavor.CrashlyticsImpl
import kotlin.reflect.KClass

class PreferenceCache(val context: Context) {
    // PERFORMANCE: Use HashSet for O(1) contains() lookup instead of O(n) ArrayList
    val changedNamespaces = HashSet<String>()
    val cache: HashMap<String, Any> = hashMapOf()
    var selectedNamespace: String? = null

    // PERFORMANCE: Throttle Crashlytics logging to reduce CPU overhead
    private var lastCrashlyticsLogTime = 0L
    private val crashlyticsLogIntervalMs = 5000L  // Log at most every 5 seconds

    fun select(namespace: String) {
        selectedNamespace = namespace
    }

    fun clear() {
        try {
            cache.clear()
        }
        catch (_: Exception) {}
    }

    @Suppress("UNCHECKED_CAST")
    fun <T : Any> get(@StringRes nameRes: Int, default: T, type: KClass<T>): T {
        if(selectedNamespace == null)
            throw IllegalStateException("No active namespace selected")

        val name = context.getString(nameRes)
        val current = uncachedGet(context, selectedNamespace!!, nameRes, default, type)
        val unchanged = cache.containsKey(name) && cache[name] == current
        if(!unchanged && !changedNamespaces.contains(selectedNamespace)) {
            selectedNamespace?.let {
                changedNamespaces.add(it)
            }
        }

        // PERFORMANCE: Only log to Crashlytics periodically to reduce overhead
        val now = System.currentTimeMillis()
        if (now - lastCrashlyticsLogTime > crashlyticsLogIntervalMs) {
            CrashlyticsImpl.setCustomKey("dsp_$name", current.toString())
            lastCrashlyticsLogTime = now
        }
        cache[name] = current as Any
        return current
    }

    inline fun <reified T : Any> get(@StringRes nameRes: Int, default: T) =
        get(nameRes, default, T::class)

    fun markChangesAsCommitted() {
        changedNamespaces.clear()
    }

    companion object {
        // PERFORMANCE: MODE_MULTI_PROCESS is deprecated and removed in API 35+
        // Using MODE_PRIVATE is safer and more efficient for single-process access
        fun getPreferences(
            context: Context,
            namespace: String,
        ): SharedPreferences {
            return context.getSharedPreferences(namespace, Context.MODE_PRIVATE)
        }

        @Suppress("UNCHECKED_CAST")
        fun <T : Any> uncachedGet(
            context: Context,
            namespace: String,
            @StringRes nameRes: Int,
            default: T,
            type: KClass<T>
        ): T {
            val name = context.getString(nameRes)
            val prefs = getPreferences(context, namespace)
            val current: T = when(type) {
                Boolean::class -> prefs.getBoolean(name, default as Boolean) as T
                String::class -> prefs.getString(name, default as String) as T
                Int::class -> prefs.getInt(name, default as Int) as T
                Float::class -> prefs.getFloat(name, default as Float) as T
                else -> throw IllegalArgumentException("Unknown type")
            }
            return current
        }

        inline fun <reified T : Any> uncachedGet(
            context: Context,
            namespace: String,
            @StringRes nameRes: Int,
            default: T
        ) = uncachedGet(context, namespace, nameRes, default, T::class)
    }
}