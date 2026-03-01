package me.timschneeberger.rootlessjamesdsp.model.preset

import android.content.Context
import android.content.SharedPreferences
import me.timschneeberger.rootlessjamesdsp.R
import me.timschneeberger.rootlessjamesdsp.utils.Constants

/**
 * S24 ULTRA - Factory Presets
 *
 * Pre-configured DSP settings for common use cases.
 * Each preset is optimized for specific audio scenarios.
 */
object FactoryPresets {

    // Preset categories
    enum class Category {
        MUSIC,
        GAMING,
        MOVIE,
        PODCAST,
        AUDIOPHILE,
        BASS_HEAD,
        SPATIAL
    }

    data class FactoryPreset(
        val id: String,
        val nameResId: Int,
        val descriptionResId: Int,
        val category: Category,
        val settings: Map<String, Any>
    )

    // ========================================================================
    // FACTORY PRESET DEFINITIONS
    // ========================================================================

    val presets = listOf(
        // ---------------------------------------------------------------------
        // MUSIC PRESETS
        // ---------------------------------------------------------------------
        FactoryPreset(
            id = "music_balanced",
            nameResId = R.string.preset_music_balanced,
            descriptionResId = R.string.preset_music_balanced_desc,
            category = Category.MUSIC,
            settings = mapOf(
                // EQ: Slight bass and treble boost
                "eq_enable" to true,
                "eq_bands" to "25.0;40.0;63.0;100.0;160.0;250.0;400.0;630.0;1000.0;1600.0;2500.0;4000.0;6300.0;10000.0;16000.0;2.0;1.5;1.0;0.5;0.0;0.0;0.0;0.0;0.0;0.0;0.5;1.0;1.5;2.0;2.5",
                // Bass boost: Light
                "bass_enable" to true,
                "bass_max_gain" to 4,
                // Stereo: Slight widening
                "stereowide_enable" to true,
                "stereowide_depth" to 60
            )
        ),

        FactoryPreset(
            id = "music_vocal_boost",
            nameResId = R.string.preset_music_vocal,
            descriptionResId = R.string.preset_music_vocal_desc,
            category = Category.MUSIC,
            settings = mapOf(
                "eq_enable" to true,
                // Boost 1-4kHz for vocals
                "eq_bands" to "25.0;40.0;63.0;100.0;160.0;250.0;400.0;630.0;1000.0;1600.0;2500.0;4000.0;6300.0;10000.0;16000.0;0.0;0.0;0.0;0.0;-1.0;-1.0;0.0;1.0;2.5;3.0;2.5;1.5;0.5;0.0;0.0",
                "stereowide_enable" to true,
                "stereowide_center" to 70  // Keep center (vocals) prominent
            )
        ),

        FactoryPreset(
            id = "music_electronic",
            nameResId = R.string.preset_music_electronic,
            descriptionResId = R.string.preset_music_electronic_desc,
            category = Category.MUSIC,
            settings = mapOf(
                "bass_enable" to true,
                "bass_max_gain" to 8,
                "bass_harmonic_enable" to true,
                "bass_harmonic_level" to 40,
                "eq_enable" to true,
                // V-shaped curve for EDM
                "eq_bands" to "25.0;40.0;63.0;100.0;160.0;250.0;400.0;630.0;1000.0;1600.0;2500.0;4000.0;6300.0;10000.0;16000.0;4.0;3.5;3.0;2.0;1.0;0.0;-1.0;-1.5;-1.0;0.0;1.0;2.0;3.0;3.5;4.0",
                "stereowide_enable" to true,
                "stereowide_depth" to 80
            )
        ),

        // ---------------------------------------------------------------------
        // GAMING PRESETS
        // ---------------------------------------------------------------------
        FactoryPreset(
            id = "gaming_fps",
            nameResId = R.string.preset_gaming_fps,
            descriptionResId = R.string.preset_gaming_fps_desc,
            category = Category.GAMING,
            settings = mapOf(
                // Boost footsteps and spatial awareness
                "eq_enable" to true,
                "eq_bands" to "25.0;40.0;63.0;100.0;160.0;250.0;400.0;630.0;1000.0;1600.0;2500.0;4000.0;6300.0;10000.0;16000.0;-2.0;-1.5;-1.0;0.0;0.0;0.5;1.0;2.0;3.0;3.5;3.0;2.0;1.0;0.5;0.0",
                // Enhance spatial positioning
                "crossfeed_enable" to false,  // Keep stereo separation
                "stereowide_enable" to true,
                "stereowide_depth" to 40,  // Subtle widening
                // Compressor to hear quiet sounds
                "compander_enable" to true
            )
        ),

        FactoryPreset(
            id = "gaming_immersive",
            nameResId = R.string.preset_gaming_immersive,
            descriptionResId = R.string.preset_gaming_immersive_desc,
            category = Category.GAMING,
            settings = mapOf(
                "bass_enable" to true,
                "bass_max_gain" to 6,
                "reverb_enable" to true,
                "reverb_preset" to 7,  // Small room
                "stereowide_enable" to true,
                "stereowide_depth" to 70
            )
        ),

        // ---------------------------------------------------------------------
        // MOVIE PRESETS
        // ---------------------------------------------------------------------
        FactoryPreset(
            id = "movie_dialog",
            nameResId = R.string.preset_movie_dialog,
            descriptionResId = R.string.preset_movie_dialog_desc,
            category = Category.MOVIE,
            settings = mapOf(
                // Boost dialog frequencies
                "eq_enable" to true,
                "eq_bands" to "25.0;40.0;63.0;100.0;160.0;250.0;400.0;630.0;1000.0;1600.0;2500.0;4000.0;6300.0;10000.0;16000.0;-2.0;-1.5;-1.0;0.0;0.5;1.0;1.5;2.0;3.0;3.5;3.0;2.0;1.0;0.0;-1.0",
                // Compress dynamic range for night viewing
                "compander_enable" to true,
                "stereowide_enable" to true,
                "stereowide_center" to 80
            )
        ),

        FactoryPreset(
            id = "movie_cinematic",
            nameResId = R.string.preset_movie_cinematic,
            descriptionResId = R.string.preset_movie_cinematic_desc,
            category = Category.MOVIE,
            settings = mapOf(
                "bass_enable" to true,
                "bass_max_gain" to 5,
                "reverb_enable" to true,
                "reverb_preset" to 5,  // Large hall
                "stereowide_enable" to true,
                "stereowide_depth" to 70
            )
        ),

        // ---------------------------------------------------------------------
        // PODCAST / AUDIOBOOK PRESETS
        // ---------------------------------------------------------------------
        FactoryPreset(
            id = "podcast_clarity",
            nameResId = R.string.preset_podcast_clarity,
            descriptionResId = R.string.preset_podcast_clarity_desc,
            category = Category.PODCAST,
            settings = mapOf(
                // Voice clarity EQ
                "eq_enable" to true,
                "eq_bands" to "25.0;40.0;63.0;100.0;160.0;250.0;400.0;630.0;1000.0;1600.0;2500.0;4000.0;6300.0;10000.0;16000.0;-4.0;-3.0;-2.0;-1.0;0.0;1.0;1.5;2.0;2.5;3.0;2.5;2.0;1.0;0.0;-1.0",
                // Compression for consistent volume
                "compander_enable" to true,
                "stereowide_enable" to false,
                "bass_enable" to false
            )
        ),

        // ---------------------------------------------------------------------
        // AUDIOPHILE PRESETS
        // ---------------------------------------------------------------------
        FactoryPreset(
            id = "audiophile_pure",
            nameResId = R.string.preset_audiophile_pure,
            descriptionResId = R.string.preset_audiophile_pure_desc,
            category = Category.AUDIOPHILE,
            settings = mapOf(
                // Minimal processing
                "eq_enable" to false,
                "bass_enable" to false,
                "stereowide_enable" to false,
                "reverb_enable" to false,
                "compander_enable" to false,
                // Only tube warmth
                "tube_enable" to true,
                "tube_drive" to 10,
                "tube_even" to 30,
                "tube_odd" to 10
            )
        ),

        FactoryPreset(
            id = "audiophile_analog",
            nameResId = R.string.preset_audiophile_analog,
            descriptionResId = R.string.preset_audiophile_analog_desc,
            category = Category.AUDIOPHILE,
            settings = mapOf(
                "tube_enable" to true,
                "tube_drive" to 20,
                "tube_even" to 50,
                "tube_odd" to 20,
                // Slight treble rolloff for analog feel
                "eq_enable" to true,
                "eq_bands" to "25.0;40.0;63.0;100.0;160.0;250.0;400.0;630.0;1000.0;1600.0;2500.0;4000.0;6300.0;10000.0;16000.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;-0.5;-1.0;-1.5;-2.0;-2.5;-3.0"
            )
        ),

        // ---------------------------------------------------------------------
        // BASS HEAD PRESETS
        // ---------------------------------------------------------------------
        FactoryPreset(
            id = "bass_extreme",
            nameResId = R.string.preset_bass_extreme,
            descriptionResId = R.string.preset_bass_extreme_desc,
            category = Category.BASS_HEAD,
            settings = mapOf(
                "bass_enable" to true,
                "bass_max_gain" to 12,
                "bass_harmonic_enable" to true,
                "bass_harmonic_level" to 60,
                "eq_enable" to true,
                // Heavy bass boost
                "eq_bands" to "25.0;40.0;63.0;100.0;160.0;250.0;400.0;630.0;1000.0;1600.0;2500.0;4000.0;6300.0;10000.0;16000.0;6.0;5.0;4.0;3.0;2.0;1.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0"
            )
        ),

        FactoryPreset(
            id = "bass_sub",
            nameResId = R.string.preset_bass_sub,
            descriptionResId = R.string.preset_bass_sub_desc,
            category = Category.BASS_HEAD,
            settings = mapOf(
                "bass_enable" to true,
                "bass_max_gain" to 10,
                "bass_manual_mode" to true,
                "bass_manual_freq" to 40,  // Focus on sub-bass
                "bass_harmonic_enable" to true,
                "bass_harmonic_level" to 70
            )
        ),

        // ---------------------------------------------------------------------
        // SPATIAL PRESETS
        // ---------------------------------------------------------------------
        FactoryPreset(
            id = "spatial_wide",
            nameResId = R.string.preset_spatial_wide,
            descriptionResId = R.string.preset_spatial_wide_desc,
            category = Category.SPATIAL,
            settings = mapOf(
                "stereowide_enable" to true,
                "stereowide_depth" to 100,
                "stereowide_low" to 30,   // Keep bass mono
                "stereowide_mid" to 100,
                "stereowide_high" to 100,
                "reverb_enable" to true,
                "reverb_preset" to 9  // Medium room
            )
        ),

        FactoryPreset(
            id = "spatial_headphones",
            nameResId = R.string.preset_spatial_headphones,
            descriptionResId = R.string.preset_spatial_headphones_desc,
            category = Category.SPATIAL,
            settings = mapOf(
                "crossfeed_enable" to true,
                "crossfeed_mode" to 1,  // BS2B level 2
                "stereowide_enable" to true,
                "stereowide_depth" to 50,
                "stereowide_center" to 60
            )
        )
    )

    /**
     * Get all presets in a specific category
     */
    fun getByCategory(category: Category): List<FactoryPreset> {
        return presets.filter { it.category == category }
    }

    /**
     * Get preset by ID
     */
    fun getById(id: String): FactoryPreset? {
        return presets.find { it.id == id }
    }

    /**
     * Apply a factory preset to the app's SharedPreferences
     */
    fun apply(context: Context, preset: FactoryPreset) {
        preset.settings.forEach { (key, value) ->
            val prefName = getPrefNameForKey(key)
            val prefs = context.getSharedPreferences(prefName, Context.MODE_PRIVATE)
            val editor = prefs.edit()

            when (value) {
                is Boolean -> editor.putBoolean(key, value)
                is Int -> editor.putInt(key, value)
                is Float -> editor.putFloat(key, value)
                is String -> editor.putString(key, value)
                is Long -> editor.putLong(key, value)
            }

            editor.apply()
        }
    }

    /**
     * Map preference key to preference file name
     */
    private fun getPrefNameForKey(key: String): String {
        return when {
            key.startsWith("eq_") -> Constants.PREF_EQ
            key.startsWith("geq_") -> Constants.PREF_GEQ
            key.startsWith("bass_") -> Constants.PREF_BASS
            key.startsWith("tube_") -> Constants.PREF_TUBE
            key.startsWith("reverb_") -> Constants.PREF_REVERB
            key.startsWith("stereowide_") -> Constants.PREF_STEREOWIDE
            key.startsWith("crossfeed_") -> Constants.PREF_CROSSFEED
            key.startsWith("compander_") -> Constants.PREF_COMPANDER
            key.startsWith("convolver_") -> Constants.PREF_CONVOLVER
            key.startsWith("ddc_") -> Constants.PREF_DDC
            key.startsWith("liveprog_") -> Constants.PREF_LIVEPROG
            key.startsWith("output_") -> Constants.PREF_OUTPUT
            else -> Constants.PREF_APP
        }
    }
}
