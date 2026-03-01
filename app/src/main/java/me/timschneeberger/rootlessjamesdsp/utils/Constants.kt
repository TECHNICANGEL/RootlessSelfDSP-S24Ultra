package me.timschneeberger.rootlessjamesdsp.utils

import me.timschneeberger.rootlessjamesdsp.BuildConfig

object Constants {
    // App-relevant preference namespaces
    const val PREF_APP = "application"
    const val PREF_VAR = "variable"



    const val ACTION_WIDGET_CLICK = "me.timschneeberger.rootlessjamesdsp.ACTION_WIDGET_CLICK"

    // DSP-relevant preference namespaces
    const val PREF_BASS = "dsp_bass"
    const val PREF_COMPANDER = "dsp_compander"
    const val PREF_CONVOLVER = "dsp_convolver"
    const val PREF_CROSSFEED = "dsp_crossfeed"
    const val PREF_DDC = "dsp_ddc"
    const val PREF_EQ = "dsp_equalizer"
    const val PREF_GEQ = "dsp_graphiceq"
    const val PREF_LIVEPROG = "dsp_liveprog"
    const val PREF_OUTPUT = "dsp_output_control"
    const val PREF_REVERB = "dsp_reverb"
    const val PREF_STEREOWIDE = "dsp_stereowide"
    const val PREF_TUBE = "dsp_tube"
    const val PREF_SPATIAL = "dsp_spatial"
    const val PREF_SEPARATION = "dsp_separation"

    // Default string values
    const val DEFAULT_CONVOLVER_ADVIMP = "-80;-100;0;0;0;0"
    const val DEFAULT_GEQ = "GraphicEQ: "
    const val DEFAULT_GEQ_INTERNAL = "GraphicEQ: 0.0 0.0;"
    const val DEFAULT_EQ = "25.0;40.0;63.0;100.0;160.0;250.0;400.0;630.0;1000.0;1600.0;2500.0;4000.0;6300.0;10000.0;16000.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0"

    // Intent actions
    const val ACTION_PREFERENCES_UPDATED = BuildConfig.APPLICATION_ID + ".action.preferences.UPDATED"
    const val ACTION_SAMPLE_RATE_UPDATED = BuildConfig.APPLICATION_ID + ".action.sample_rate.UPDATED"
    const val ACTION_PRESET_LOADED = BuildConfig.APPLICATION_ID + ".action.preset.LOADED"
    const val ACTION_GRAPHIC_EQ_CHANGED = BuildConfig.APPLICATION_ID + ".action.preferences.graphiceq.CHANGED"
    const val ACTION_SESSION_CHANGED = BuildConfig.APPLICATION_ID + ".action.session.CHANGED"
    const val ACTION_SERVICE_STARTED = BuildConfig.APPLICATION_ID + ".action.service.STARTED"
    const val ACTION_SERVICE_STOPPED = BuildConfig.APPLICATION_ID + ".action.service.STOPPED"
    const val ACTION_SERVICE_RELOAD_LIVEPROG = BuildConfig.APPLICATION_ID + ".action.service.RELOAD_LIVEPROG"
    const val ACTION_SERVICE_HARD_REBOOT_CORE = BuildConfig.APPLICATION_ID + ".action.service.HARD_REBOOT_CORE"
    const val ACTION_SERVICE_SOFT_REBOOT_CORE = BuildConfig.APPLICATION_ID + ".action.service.SOFT_REBOOT_CORE"
    const val ACTION_PROCESSOR_MESSAGE = BuildConfig.APPLICATION_ID + ".action.service.PROCESSOR_MESSAGE"
    const val ACTION_DISCARD_AUTHORIZATION = BuildConfig.APPLICATION_ID + ".action.service.DISCARD_AUTHORIZATION"
    const val ACTION_REPORT_SAMPLE_RATE = BuildConfig.APPLICATION_ID + ".action.service.REPORT_SAMPLE_RATE"
    const val ACTION_BACKUP_RESTORED = BuildConfig.APPLICATION_ID + ".action.backup.RESTORED"

    // Intent extras
    const val EXTRA_SAMPLE_RATE = BuildConfig.APPLICATION_ID + ".extra.service.SAMPLE_RATE"

    // S24 ULTRA - Visualization broadcasts
    const val ACTION_VISUALIZER_REQUEST = BuildConfig.APPLICATION_ID + ".action.visualizer.REQUEST"
    const val ACTION_VISUALIZER_DATA = BuildConfig.APPLICATION_ID + ".action.visualizer.DATA"
    const val EXTRA_VU_LEFT_RMS = "vu_left_rms"
    const val EXTRA_VU_RIGHT_RMS = "vu_right_rms"
    const val EXTRA_VU_LEFT_PEAK = "vu_left_peak"
    const val EXTRA_VU_RIGHT_PEAK = "vu_right_peak"
    const val EXTRA_LATENCY_MS = "latency_ms"
    const val EXTRA_BUFFER_FILL = "buffer_fill"
    const val EXTRA_THROUGHPUT = "throughput"

    // S24 ULTRA RAW POWER - FFT visualization
    const val ACTION_FFT_DATA = BuildConfig.APPLICATION_ID + ".action.visualizer.FFT"
    const val EXTRA_FFT_MAGNITUDES = "fft_magnitudes"
    const val EXTRA_FFT_SIZE = "fft_size"

    // S24 ULTRA RAW POWER - Stereo FFT visualization
    const val EXTRA_FFT_LEFT_MAGNITUDES = "fft_left_magnitudes"
    const val EXTRA_FFT_RIGHT_MAGNITUDES = "fft_right_magnitudes"
    const val EXTRA_FFT_STEREO_MODE = "fft_stereo_mode"

    // S24 ULTRA - Stem level visualization
    const val ACTION_STEM_LEVELS = BuildConfig.APPLICATION_ID + ".action.visualizer.STEM_LEVELS"
    const val EXTRA_STEM_VOCALS_LEVEL = "stem_vocals_level"
    const val EXTRA_STEM_DRUMS_LEVEL = "stem_drums_level"
    const val EXTRA_STEM_BASS_LEVEL = "stem_bass_level"
    const val EXTRA_STEM_OTHER_LEVEL = "stem_other_level"

    // S24 ULTRA - Hardware Accelerator ready (NPU/GPU initialized)
    const val ACTION_HARDWARE_ACCELERATOR_READY = BuildConfig.APPLICATION_ID + ".action.hardware.READY"
    const val EXTRA_SOURCE_SEPARATION_AVAILABLE = "source_separation_available"
    const val EXTRA_SPATIAL_AUDIO_AVAILABLE = "spatial_audio_available"

    // S24 ULTRA - Separation state updates (buffering, processing, playing)
    const val ACTION_SEPARATION_STATE = BuildConfig.APPLICATION_ID + ".action.separation.STATE"
    const val EXTRA_SEPARATION_STATE = "separation_state"
    const val EXTRA_SEPARATION_PROGRESS = "separation_progress"
    const val EXTRA_SEPARATION_MESSAGE = "separation_message"
    const val EXTRA_NPU_USAGE = "npu_usage"

    // S24 ULTRA - Pipeline timing stats
    const val ACTION_PIPELINE_STATS = BuildConfig.APPLICATION_ID + ".action.pipeline.STATS"
    const val EXTRA_BUFFER_PCT = "buffer_pct"
    const val EXTRA_STFT_MS = "stft_ms"
    const val EXTRA_NPU_MS = "npu_ms"
    const val EXTRA_ISTFT_MS = "istft_ms"
    const val EXTRA_QUEUE_SIZE = "queue_size"

    // S24 ULTRA - Individual model timing stats
    const val ACTION_MODEL_STATS = BuildConfig.APPLICATION_ID + ".action.model.STATS"
    const val EXTRA_VOCALS_MS = "vocals_ms"
    const val EXTRA_DRUMS_MS = "drums_ms"
    const val EXTRA_BASS_MS = "bass_ms"
    const val EXTRA_OTHER_MS = "other_ms"
    const val EXTRA_TOTAL_MS = "total_ms"
    const val EXTRA_USING_NPU = "using_npu"
    const val EXTRA_BACKEND_NAME = "backend_name"
}