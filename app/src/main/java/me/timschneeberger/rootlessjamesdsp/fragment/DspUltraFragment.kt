package me.timschneeberger.rootlessjamesdsp.fragment

import android.bluetooth.BluetoothA2dp
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothCodecConfig
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.viewpager2.adapter.FragmentStateAdapter
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator
import me.timschneeberger.rootlessjamesdsp.R
import me.timschneeberger.rootlessjamesdsp.utils.Constants
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.registerLocalReceiver
import me.timschneeberger.rootlessjamesdsp.utils.extensions.ContextExtensions.unregisterLocalReceiver
import me.timschneeberger.rootlessjamesdsp.utils.preferences.Preferences
import org.koin.android.ext.android.inject
import timber.log.Timber

/**
 * S24 ULTRA - Tab-based DSP Dashboard
 *
 * Tabs:
 * 1. Dashboard - Visualizers & Stats
 * 2. Separation - Source Separation
 * 3. Spatial - Spatial Audio & Head Tracking
 * 4. Effects - DSP Effects
 */
class DspUltraFragment : Fragment(), SharedPreferences.OnSharedPreferenceChangeListener {

    private val prefsApp: Preferences.App by inject()

    // Views
    private var tabLayout: TabLayout? = null
    private var viewPager: ViewPager2? = null
    private var fabDspToggle: FloatingActionButton? = null
    private var statusText: TextView? = null
    private var bluetoothContainer: LinearLayout? = null
    private var bluetoothDeviceName: TextView? = null
    private var bluetoothCodec: TextView? = null

    // Tab fragments references
    private var separationTab: SourceSeparationTabFragment? = null
    private var spatialTab: SpatialAudioTabFragment? = null

    // Bluetooth A2DP for real codec detection
    private var bluetoothA2dp: BluetoothA2dp? = null
    private var bluetoothAdapter: BluetoothAdapter? = null

    // State
    private var isServiceActive = false
    private var sourceSeparationEnabled = false
    private var spatialAudioEnabled = false

    private val serviceReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (!isAdded || view == null) return

            when (intent.action) {
                Constants.ACTION_SERVICE_STARTED -> {
                    isServiceActive = true
                    statusText?.text = "● ACTIVE"
                    context?.let { ctx ->
                        statusText?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_success))
                    }
                    performHapticFeedback(true)
                }
                Constants.ACTION_SERVICE_STOPPED -> {
                    isServiceActive = false
                    statusText?.text = "○ STANDBY"
                    context?.let { ctx ->
                        statusText?.setTextColor(ContextCompat.getColor(ctx, R.color.ultra_text_tertiary))
                    }
                    performHapticFeedback(false)
                }
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_dsp_ultra, container, false)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefsApp.registerOnSharedPreferenceChangeListener(this)
    }

    override fun onDestroy() {
        prefsApp.unregisterOnSharedPreferenceChangeListener(this)
        super.onDestroy()
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Find views
        tabLayout = view.findViewById(R.id.tab_layout)
        viewPager = view.findViewById(R.id.view_pager)
        fabDspToggle = view.findViewById(R.id.fab_dsp_toggle)
        statusText = view.findViewById(R.id.status_text)
        bluetoothContainer = view.findViewById(R.id.bluetooth_container)
        bluetoothDeviceName = view.findViewById(R.id.bluetooth_device_name)
        bluetoothCodec = view.findViewById(R.id.bluetooth_codec)

        setupViewPager()
        setupFab()
        updateBluetoothDevice()

        // Initialize FAB state from preferences
        // Child fragment callbacks will also update this, but we initialize here for immediate UI feedback
        sourceSeparationEnabled = prefsApp.get<Boolean>(R.string.key_source_separation_enable, false)
        spatialAudioEnabled = prefsApp.get<Boolean>(R.string.key_spatial_audio_enable, false)
        updateFabState()

        // Initial FAB visibility based on default tab (Dashboard = 0, so FAB hidden initially)
        updateFabVisibility(0)

        Timber.i("S24 ULTRA: DspUltraFragment initialized with tabs")
    }

    private fun setupViewPager() {
        val vp = viewPager ?: return
        val tl = tabLayout ?: return

        // Create adapter
        vp.adapter = DspUltraPagerAdapter(this)
        vp.offscreenPageLimit = 2  // Keep adjacent tabs in memory

        // Link TabLayout with ViewPager2
        TabLayoutMediator(tl, vp) { tab, position ->
            tab.text = when (position) {
                0 -> "DASHBOARD"
                1 -> "SEPARATION"
                2 -> "SPATIAL"
                3 -> "EFFECTS"
                else -> ""
            }
        }.attach()

        // Listen for tab changes to update FAB
        vp.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                updateFabVisibility(position)
            }
        })
    }

    private fun updateFabVisibility(position: Int) {
        // Show FAB only on separation and spatial tabs
        fabDspToggle?.isVisible = position == 1 || position == 2
    }

    private fun setupFab() {
        fabDspToggle?.setOnClickListener {
            // Toggle both separation and spatial
            val anyEnabled = sourceSeparationEnabled || spatialAudioEnabled
            val newState = !anyEnabled

            // Toggle via preferences - child fragments listen for preference changes
            // This approach works even if fragment references are null after configuration change
            prefsApp.set(R.string.key_source_separation_enable, newState)
            prefsApp.set(R.string.key_spatial_audio_enable, newState)

            // Also update via fragment references if available (for immediate UI feedback)
            separationTab?.setEnabled(newState)
            spatialTab?.setEnabled(newState)

            performHapticFeedback(newState)
        }

        updateFabState()
    }

    private fun updateFabState() {
        val ctx = context ?: return
        val anyEnabled = sourceSeparationEnabled || spatialAudioEnabled

        if (anyEnabled) {
            fabDspToggle?.setImageResource(R.drawable.ic_dsp_power)
            fabDspToggle?.imageTintList = ContextCompat.getColorStateList(ctx, R.color.ultra_success)
            fabDspToggle?.backgroundTintList = ContextCompat.getColorStateList(ctx, R.color.ultra_glass_bg)
        } else {
            fabDspToggle?.setImageResource(R.drawable.ic_dsp_off)
            fabDspToggle?.imageTintList = ContextCompat.getColorStateList(ctx, R.color.ultra_text_tertiary)
            fabDspToggle?.backgroundTintList = ContextCompat.getColorStateList(ctx, R.color.ultra_glass_bg)
        }

        // Bounce animation
        fabDspToggle?.animate()
            ?.scaleX(1.15f)
            ?.scaleY(1.15f)
            ?.setDuration(80)
            ?.withEndAction {
                fabDspToggle?.animate()
                    ?.scaleX(1f)
                    ?.scaleY(1f)
                    ?.setDuration(100)
                    ?.start()
            }
            ?.start()
    }

    fun onSeparationStateChanged(enabled: Boolean) {
        sourceSeparationEnabled = enabled
        updateFabState()
    }

    fun onSpatialStateChanged(enabled: Boolean) {
        spatialAudioEnabled = enabled
        updateFabState()
    }

    override fun onSharedPreferenceChanged(sharedPreferences: SharedPreferences?, key: String?) {
        // Handle preference changes if needed
    }

    override fun onResume() {
        super.onResume()

        context?.registerLocalReceiver(serviceReceiver, IntentFilter().apply {
            addAction(Constants.ACTION_SERVICE_STARTED)
            addAction(Constants.ACTION_SERVICE_STOPPED)
        })

        updateBluetoothDevice()
    }

    override fun onPause() {
        super.onPause()
        context?.unregisterLocalReceiver(serviceReceiver)
    }

    override fun onDestroyView() {
        // MEMORY LEAK FIX: Ensure receiver is unregistered even if onPause didn't fire
        try {
            context?.unregisterLocalReceiver(serviceReceiver)
        } catch (_: Exception) { }

        // MEMORY LEAK FIX: Close BluetoothA2dp proxy
        try {
            bluetoothAdapter?.closeProfileProxy(BluetoothProfile.A2DP, bluetoothA2dp)
        } catch (_: Exception) { }
        bluetoothA2dp = null
        bluetoothAdapter = null

        // Cancel any running FAB animations
        fabDspToggle?.animate()?.cancel()
        fabDspToggle?.setOnClickListener(null)

        tabLayout = null
        viewPager = null
        fabDspToggle = null
        statusText = null
        bluetoothContainer = null
        bluetoothDeviceName = null
        bluetoothCodec = null
        separationTab = null
        spatialTab = null

        super.onDestroyView()
    }

    // Bluetooth A2DP Profile Listener for real codec detection
    private val bluetoothProfileListener = object : BluetoothProfile.ServiceListener {
        override fun onServiceConnected(profile: Int, proxy: BluetoothProfile) {
            if (profile == BluetoothProfile.A2DP) {
                bluetoothA2dp = proxy as BluetoothA2dp
                Timber.d("BluetoothA2dp service connected")
                // Update codec info now that we have the A2DP proxy
                updateBluetoothCodec()
            }
        }

        override fun onServiceDisconnected(profile: Int) {
            if (profile == BluetoothProfile.A2DP) {
                bluetoothA2dp = null
                Timber.d("BluetoothA2dp service disconnected")
            }
        }
    }

    @Suppress("MissingPermission")
    private fun initBluetoothA2dp() {
        val ctx = context ?: return
        try {
            val bluetoothManager = ctx.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
            bluetoothAdapter = bluetoothManager?.adapter
            bluetoothAdapter?.getProfileProxy(ctx, bluetoothProfileListener, BluetoothProfile.A2DP)
        } catch (e: Exception) {
            Timber.w(e, "Failed to initialize Bluetooth A2DP proxy")
        }
    }

    @Suppress("MissingPermission")
    private fun updateBluetoothDevice() {
        val ctx = context ?: return

        try {
            val audioManager = ctx.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return

            val devices = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
            val btDevice = devices.firstOrNull {
                it.type == AudioDeviceInfo.TYPE_BLUETOOTH_A2DP ||
                it.type == AudioDeviceInfo.TYPE_BLE_HEADSET ||
                it.type == AudioDeviceInfo.TYPE_BLE_SPEAKER
            }

            if (btDevice != null) {
                bluetoothContainer?.isVisible = true
                val name = btDevice.productName?.toString() ?: "BT Audio"
                // Shorten name if too long
                bluetoothDeviceName?.text = if (name.length > 20) name.take(20) + "…" else name

                Timber.i("BT device: ${btDevice.productName}")

                // Initialize A2DP proxy if not already done
                if (bluetoothA2dp == null) {
                    initBluetoothA2dp()
                    // Codec will be updated when service connects
                    bluetoothCodec?.text = "..."
                } else {
                    updateBluetoothCodec()
                }
            } else {
                bluetoothContainer?.isVisible = false
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to detect Bluetooth device")
            bluetoothContainer?.isVisible = false
        }
    }

    @Suppress("MissingPermission")
    private fun updateBluetoothCodec() {
        val a2dp = bluetoothA2dp ?: return

        try {
            // Get connected A2DP devices
            val connectedDevices = a2dp.connectedDevices
            if (connectedDevices.isEmpty()) {
                bluetoothCodec?.text = "N/A"
                return
            }

            val device = connectedDevices.firstOrNull() ?: return

            // Get codec status using reflection (hidden API)
            // These APIs exist but are not public, so we access them via reflection
            try {
                val getCodecStatusMethod = BluetoothA2dp::class.java.getMethod("getCodecStatus", android.bluetooth.BluetoothDevice::class.java)
                val codecStatus = getCodecStatusMethod.invoke(a2dp, device)

                if (codecStatus != null) {
                    val getCodecConfigMethod = codecStatus.javaClass.getMethod("getCodecConfig")
                    val codecConfig = getCodecConfigMethod.invoke(codecStatus)

                    if (codecConfig != null) {
                        val codecTypeField = codecConfig.javaClass.getDeclaredField("mCodecType")
                        codecTypeField.isAccessible = true
                        val codecType = codecTypeField.getInt(codecConfig)

                        val sampleRateField = codecConfig.javaClass.getDeclaredField("mSampleRate")
                        sampleRateField.isAccessible = true
                        val sampleRate = sampleRateField.getInt(codecConfig)

                        val bitsPerSampleField = codecConfig.javaClass.getDeclaredField("mBitsPerSample")
                        bitsPerSampleField.isAccessible = true
                        val bitsPerSample = bitsPerSampleField.getInt(codecConfig)

                        val codecName = getCodecName(codecType)
                        val sampleRateStr = getSampleRateString(sampleRate)
                        val bitsPerSampleStr = getBitsPerSampleString(bitsPerSample)

                        // Format: "LDAC 96kHz/24bit" or just "LDAC" if details unavailable
                        val codecText = buildString {
                            append(codecName)
                            if (sampleRateStr != null || bitsPerSampleStr != null) {
                                append(" ")
                                if (sampleRateStr != null) append(sampleRateStr)
                                if (bitsPerSampleStr != null) append("/$bitsPerSampleStr")
                            }
                        }
                        bluetoothCodec?.text = codecText
                        Timber.i("Real BT codec: $codecText (type=$codecType)")
                    } else {
                        bluetoothCodec?.text = "A2DP"
                    }
                } else {
                    bluetoothCodec?.text = "A2DP"
                }
            } catch (e: Exception) {
                Timber.w(e, "Failed to get codec details via reflection, falling back to A2DP")
                bluetoothCodec?.text = "A2DP"
            }
        } catch (e: Exception) {
            Timber.w(e, "Failed to get Bluetooth codec")
            bluetoothCodec?.text = "A2DP"
        }
    }

    private fun getCodecName(codecType: Int): String {
        return when (codecType) {
            BluetoothCodecConfig.SOURCE_CODEC_TYPE_SBC -> "SBC"
            BluetoothCodecConfig.SOURCE_CODEC_TYPE_AAC -> "AAC"
            BluetoothCodecConfig.SOURCE_CODEC_TYPE_APTX -> "aptX"
            BluetoothCodecConfig.SOURCE_CODEC_TYPE_APTX_HD -> "aptX HD"
            BluetoothCodecConfig.SOURCE_CODEC_TYPE_LDAC -> "LDAC"
            BluetoothCodecConfig.SOURCE_CODEC_TYPE_LC3 -> "LC3"
            BluetoothCodecConfig.SOURCE_CODEC_TYPE_OPUS -> "Opus"
            6 -> "aptX Adaptive"  // SOURCE_CODEC_TYPE_APTX_ADAPTIVE not in all SDKs
            7 -> "aptX TWS+"      // SOURCE_CODEC_TYPE_APTX_TWSP
            1000001 -> "SSC"      // Samsung Scalable Codec
            else -> "Codec $codecType"
        }
    }

    private fun getSampleRateString(sampleRate: Int): String? {
        return when (sampleRate) {
            BluetoothCodecConfig.SAMPLE_RATE_44100 -> "44.1kHz"
            BluetoothCodecConfig.SAMPLE_RATE_48000 -> "48kHz"
            BluetoothCodecConfig.SAMPLE_RATE_88200 -> "88.2kHz"
            BluetoothCodecConfig.SAMPLE_RATE_96000 -> "96kHz"
            BluetoothCodecConfig.SAMPLE_RATE_176400 -> "176.4kHz"
            BluetoothCodecConfig.SAMPLE_RATE_192000 -> "192kHz"
            else -> null
        }
    }

    private fun getBitsPerSampleString(bitsPerSample: Int): String? {
        return when (bitsPerSample) {
            BluetoothCodecConfig.BITS_PER_SAMPLE_16 -> "16bit"
            BluetoothCodecConfig.BITS_PER_SAMPLE_24 -> "24bit"
            BluetoothCodecConfig.BITS_PER_SAMPLE_32 -> "32bit"
            else -> null
        }
    }

    @Suppress("DEPRECATION")
    private fun performHapticFeedback(isEnabled: Boolean) {
        val ctx = context ?: return

        try {
            val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vm = ctx.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                vm.defaultVibrator
            } else {
                ctx.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            }

            if (!vibrator.hasVibrator()) return

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val effect = if (isEnabled) {
                    VibrationEffect.createWaveform(longArrayOf(0, 50, 50, 50), -1)
                } else {
                    VibrationEffect.createOneShot(30, VibrationEffect.DEFAULT_AMPLITUDE)
                }
                vibrator.vibrate(effect)
            } else {
                vibrator.vibrate(50)
            }
        } catch (e: Exception) {
            Timber.w(e, "Haptic feedback failed")
        }
    }

    /**
     * ViewPager2 Adapter for DSP Ultra tabs
     */
    private inner class DspUltraPagerAdapter(fragment: Fragment) : FragmentStateAdapter(fragment) {

        override fun getItemCount(): Int = 4

        override fun createFragment(position: Int): Fragment {
            return when (position) {
                0 -> DashboardTabFragment.newInstance()
                1 -> SourceSeparationTabFragment.newInstance().also {
                    separationTab = it
                    it.onSeparationStateChanged = { enabled ->
                        onSeparationStateChanged(enabled)
                    }
                }
                2 -> SpatialAudioTabFragment.newInstance().also {
                    spatialTab = it
                    it.onSpatialStateChanged = { enabled ->
                        onSpatialStateChanged(enabled)
                    }
                }
                3 -> EffectsTabFragment.newInstance()
                else -> DashboardTabFragment.newInstance()
            }
        }
    }

    companion object {
        fun newInstance(): DspUltraFragment {
            return DspUltraFragment()
        }
    }
}
