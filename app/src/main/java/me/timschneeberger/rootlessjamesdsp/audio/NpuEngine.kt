package me.timschneeberger.rootlessjamesdsp.audio

import timber.log.Timber

/**
 * S24 ULTRA - NPU Engine Info
 *
 * Provides information about available NNAPI accelerators.
 */
object NpuEngine {

    private const val TAG = "NpuEngine"
    private var libraryLoaded = false

    init {
        try {
            System.loadLibrary("npu_audio")
            libraryLoaded = true
        } catch (e: UnsatisfiedLinkError) {
            libraryLoaded = false
            Timber.tag(TAG).w("npu_audio library not available")
        }
    }

    /**
     * Check if NNAPI is supported on this device
     */
    fun isSupported(): Boolean {
        if (!libraryLoaded) return false
        return try {
            nativeIsSupported()
        } catch (e: UnsatisfiedLinkError) {
            false
        }
    }

    /**
     * Get info about available NNAPI devices
     */
    fun getDeviceInfo(): String {
        if (!libraryLoaded) return "npu_audio library not available"
        return try {
            nativeGetDeviceInfo()
        } catch (e: UnsatisfiedLinkError) {
            "NNAPI not available"
        }
    }

    @JvmStatic
    private external fun nativeIsSupported(): Boolean

    @JvmStatic
    private external fun nativeGetDeviceInfo(): String
}
