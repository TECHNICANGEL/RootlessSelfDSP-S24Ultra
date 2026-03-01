package me.timschneeberger.rootlessjamesdsp.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.system.Os
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * KUIELab MDX-Net Source Separation via ONNX Runtime + QNN (Qualcomm AI Engine Direct)
 *
 * Uses 4 separate models with Hexagon NPU acceleration on Snapdragon 8 Gen 3:
 * - kuielab_vocals.onnx (29MB) - 2D conv U-Net
 * - kuielab_drums.onnx (21MB) - 2D conv U-Net
 * - kuielab_bass.onnx (29MB) - 2D conv U-Net
 * - kuielab_other.onnx (29MB) - 2D conv U-Net
 *
 * QNN HTP (Hexagon Tensor Processor) provides direct NPU access for maximum performance.
 * Input format: [batch=1, dim_c=4, dim_f, dim_t] where dim_c=4 is stereo complex (L_real, L_imag, R_real, R_imag)
 */
class SourceSeparator(private val context: Context) {

    enum class SeparationState {
        IDLE,           // Not processing
        BUFFERING,      // Accumulating audio into chunk
        PROCESSING_STFT,  // Running STFT transform
        PROCESSING_NPU,   // Running NPU inference (4 models)
        PROCESSING_ISTFT, // Running inverse STFT
        PLAYING,        // Playing back separated audio
        EXHAUSTED       // Queue empty, waiting for more data
    }

    interface StateListener {
        fun onStateChanged(state: SeparationState, progress: Float, message: String, npuUsage: Float)
        fun onPipelineStats(bufferPct: Int, stftMs: Int, npuMs: Int, istftMs: Int, queueSize: Int, state: SeparationState, progress: Float)
        fun onModelStats(vocalsMs: Int, drumsMs: Int, bassMs: Int, otherMs: Int, totalMs: Int, usingNpu: Boolean, backendName: String)
    }

    // Pipeline timing stats
    private var lastStftMs = 0
    private var lastNpuMs = 0
    private var lastIstftMs = 0

    // Individual model timing (for diagnostics)
    private var vocalsMs = 0
    private var drumsMs = 0
    private var bassMs = 0
    private var otherMs = 0

    /**
     * Per-model parameters extracted from actual ONNX shapes.
     * KUIELab models have different n_fft_scale per stem which affects dimensions.
     */
    data class ModelParams(
        val dimC: Int,      // Channels (should be 4 for stereo complex)
        val dimF: Int,      // Frequency bins
        val dimT: Int,      // Time frames
        val nFft: Int,      // FFT size derived from dim_f
        val hopLength: Int  // Hop length
    )

    // Model-specific parameters (populated during initialization)
    private var vocalsParams: ModelParams? = null
    private var drumsParams: ModelParams? = null
    private var bassParams: ModelParams? = null
    private var otherParams: ModelParams? = null

    companion object {
        private const val TAG = "SourceSeparator"

        // KUIELab MDX-Net COMBINED 4-stem model (single inference, 4 outputs)
        // Input: [1, 4, 2048, 256] -> Outputs: vocals, drums, bass, other [1, 4, 2048, 256] each
        private const val MODEL_COMBINED = "models/kuielab_4stem_combined.onnx"
        private const val MODEL_FP16 = "models/kuielab_4stem_fp16.onnx"
        private const val MODEL_INT8 = "models/kuielab_4stem_int8.onnx"  // INT8 quantized - NPU preferred

        // Audio parameters - AGGRESSIVE DOWNSAMPLE for NPU performance
        private const val INPUT_SAMPLE_RATE = 44100
        private const val PROCESS_SAMPLE_RATE = 16000  // 16kHz for ~8x speedup
        private const val DOWNSAMPLE_FACTOR = 3  // 44100/16000 ≈ 2.76, use 3 for simplicity
        private const val CHUNK_SECONDS = 2  // 2 seconds per chunk for faster response
        private const val CHUNK_SAMPLES = INPUT_SAMPLE_RATE * CHUNK_SECONDS
        private const val CHUNK_SAMPLES_STEREO = CHUNK_SAMPLES * 2

        // n_fft and hop_length - MINIMAL for speed
        // Using very small FFT sizes for 16kHz

        private const val MIN_INPUT_SIZE = 256
    }

    // NPU tracking
    private var lastInferenceTimeMs = 0L
    private var npuUsagePercent = 0f
    private val CHUNK_REALTIME_MS = CHUNK_SECONDS * 1000f

    // Resampling buffers
    private var downsampleBufferL = FloatArray(0)
    private var downsampleBufferR = FloatArray(0)
    private var upsampleBufferL = FloatArray(0)
    private var upsampleBufferR = FloatArray(0)

    private var stateListener: StateListener? = null
    private var currentState = SeparationState.IDLE

    fun setStateListener(listener: StateListener?) { stateListener = listener }
    fun getLastInferenceTimeMs() = lastInferenceTimeMs
    fun getNpuUsage() = npuUsagePercent
    fun getCurrentState() = currentState

    private fun updateState(state: SeparationState, progress: Float, message: String) {
        currentState = state
        currentProgress = progress
        stateListener?.onStateChanged(state, progress, message, npuUsagePercent)
    }

    // Track current progress for pipeline stats
    private var currentProgress = 0f

    // Throttle UI updates (max 20 updates/sec for smooth progress)
    private var lastUiUpdateTime = 0L
    private val UI_UPDATE_INTERVAL_MS = 50L

    private fun updatePipelineStats(bufferPct: Int, queueSize: Int, forceUpdate: Boolean = false) {
        val now = System.currentTimeMillis()
        if (!forceUpdate && now - lastUiUpdateTime < UI_UPDATE_INTERVAL_MS) return
        lastUiUpdateTime = now
        stateListener?.onPipelineStats(bufferPct, lastStftMs, lastNpuMs, lastIstftMs, queueSize, currentState, currentProgress)
    }

    // ONNX Runtime - SINGLE combined session for NPU (4 outputs)
    private var ortEnvironment: OrtEnvironment? = null
    private var combinedSession: OrtSession? = null

    // Combined model params (unified dimensions)
    private var combinedParams: ModelParams? = null

    private val isInitialized = AtomicBoolean(false)

    // Stem levels
    private val vocalsLevel = AtomicReference(1.0f)
    private val drumsLevel = AtomicReference(1.0f)
    private val bassLevel = AtomicReference(1.0f)
    private val otherLevel = AtomicReference(1.0f)

    private val processLock = Any()

    // Buffers
    private var vocalsBuffer = FloatArray(0)
    private var drumsBuffer = FloatArray(0)
    private var bassBuffer = FloatArray(0)
    private var otherBuffer = FloatArray(0)

    private var audioAccumBuffer = FloatArray(0)
    private var audioAccumPos = 0

    // Executor for kuielab inference - 2 threads for parallel chunk processing
    private val onnxExecutor = Executors.newFixedThreadPool(2) { r ->
        Thread(r, "KUIELab-NPU-Inference").apply {
            priority = Thread.MAX_PRIORITY
        }
    }

    // Executor for parallel model inference within a chunk - 4 threads for 4 models
    private val modelExecutor = Executors.newFixedThreadPool(4) { r ->
        Thread(r, "KUIELab-Model").apply {
            priority = Thread.MAX_PRIORITY
        }
    }

    private val inflightJobs = java.util.concurrent.ConcurrentLinkedQueue<Future<StemResult?>>()
    private val onnxJobCount = java.util.concurrent.atomic.AtomicInteger(0)

    private val resultQueue = java.util.concurrent.ConcurrentLinkedQueue<StemResult>()
    private var currentResult: StemResult? = null
    private var playbackPos = 0
    private val MIN_QUEUE_BEFORE_PLAYBACK = 8  // Start playback after 8 chunks (16s buffer with 2s chunks)
    private val MAX_PARALLEL_JOBS = 2  // Allow 2 parallel inference jobs
    private var hasStartedPlayback = false
    private var lastPlayedResult: StemResult? = null  // Keep last chunk for seamless fallback

    // Fade in/out for smooth transitions
    private val FADE_SAMPLES = 4410  // 100ms fade at 44.1kHz
    private var fadeOutPos = 0  // Position in fade out (0 = not fading)
    private var fadeInPos = 0   // Position in fade in (0 = not fading)
    private var isFadingOut = false
    private var isFadingIn = false

    // Passthrough buffer - preallocated to avoid GC pressure
    private var passthroughBuffer = FloatArray(8192)

    private val OVERLAP_SAMPLES = INPUT_SAMPLE_RATE  // 1 second overlap
    private var overlapBuffer = FloatArray(0)
    private val CROSSFADE = 4410  // 100ms crossfade

    // Precomputed tables - lazily initialized after model params are known
    private var hannWindows: MutableMap<Int, FloatArray> = mutableMapOf()

    private fun getHannWindow(size: Int): FloatArray {
        return hannWindows.getOrPut(size) {
            FloatArray(size) { i ->
                (0.5f * (1f - cos(2f * PI.toFloat() * i / size))).toFloat()
            }
        }
    }

    private val crossfadeTable: FloatArray by lazy {
        FloatArray(CROSSFADE) { i ->
            val t = i.toFloat() / CROSSFADE
            (1f - cos(t * PI.toFloat())) / 2f
        }
    }

    // FFT tables - keyed by FFT size since models have different sizes
    private var fftCosTable: MutableMap<Int, FloatArray> = mutableMapOf()
    private var fftSinTable: MutableMap<Int, FloatArray> = mutableMapOf()
    private var bitReversalTable: MutableMap<Int, IntArray> = mutableMapOf()

    fun initialize(sampleRate: Int = INPUT_SAMPLE_RATE): Boolean {
        Timber.tag(TAG).i("Initializing KUIELab MDX-Net COMBINED 4-stem model with QNN...")

        if (isInitialized.get()) return true

        ortEnvironment = OrtEnvironment.getEnvironment()

        // Create session options with QNN/NNAPI NPU acceleration
        val sessionOptions = createSessionOptions()

        // Always use FP32 model - FP16 requires FP16 input tensors which we don't support yet
        Timber.tag(TAG).i("Loading FP32 model...")
        combinedSession = loadModel(MODEL_COMBINED, sessionOptions)
        if (combinedSession != null) {
            Timber.tag(TAG).i("Loaded FP32 model with backend: $backendType")
        }
        combinedParams = extractModelParams(combinedSession, "combined")

        if (combinedSession == null) {
            Timber.tag(TAG).e("Failed to load combined kuielab model")
            release()
            return false
        }

        if (combinedParams == null) {
            Timber.tag(TAG).e("Failed to extract model parameters")
            release()
            return false
        }

        // Initialize FFT tables for unified params
        initFftTablesForSize(combinedParams!!.nFft)

        // Use combined params for all stems (unified dimensions)
        vocalsParams = combinedParams
        drumsParams = combinedParams
        bassParams = combinedParams
        otherParams = combinedParams

        // Log model info
        Timber.tag(TAG).i("=== Combined Model Parameters ===")
        Timber.tag(TAG).i("Input:  dim_c=${combinedParams!!.dimC}, dim_f=${combinedParams!!.dimF}, dim_t=${combinedParams!!.dimT}, n_fft=${combinedParams!!.nFft}")
        Timber.tag(TAG).i("Output: vocals, drums, bass, other (4 stems)")

        isInitialized.set(true)
        Timber.tag(TAG).i("KUIELab MDX-Net initialized ($backendType)")
        return true
    }

    /**
     * Extract model parameters from ONNX input shape.
     * KUIELab models expect [batch=1, dim_c=4, dim_f, dim_t]
     *
     * KUIElab n_fft_scale values:
     * - bass: 8 → n_fft = 16384
     * - drums: 2 → n_fft = 4096
     * - vocals: 3 → n_fft = 6144
     * - other: 4 → n_fft = 8192
     */
    private fun extractModelParams(session: OrtSession?, name: String): ModelParams? {
        if (session == null) return null

        return try {
            val inputInfo = session.inputInfo.values.first()
            val tensorInfo = inputInfo.info as? ai.onnxruntime.TensorInfo ?: return null
            val shape = tensorInfo.shape

            if (shape.size != 4) {
                Timber.tag(TAG).e("$name model: unexpected shape rank ${shape.size}, expected 4")
                return null
            }

            val dimC = shape[1].toInt()  // Should be 4 for stereo complex
            val dimF = shape[2].toInt()  // Frequency bins (cropped)
            val dimT = shape[3].toInt()  // Time frames

            // MINIMAL n_fft for 16kHz processing - speed over quality
            // Using 512-base FFT sizes for maximum speed
            val nFftScale = when (name) {
                "bass" -> 2    // 1024 FFT
                "drums" -> 1   // 512 FFT
                "vocals" -> 1  // 512 FFT
                "other" -> 1   // 512 FFT
                else -> 1
            }
            val nFft = nFftScale * 512  // Minimal FFT for speed

            // Hop length = n_fft / 2 for 50% overlap
            val hopLength = nFft / 2

            Timber.tag(TAG).i("$name model: shape=[${shape.contentToString()}] -> dim_c=$dimC, dim_f=$dimF, dim_t=$dimT, n_fft=$nFft (scale=$nFftScale), hop=$hopLength")

            ModelParams(
                dimC = dimC,
                dimF = dimF,
                dimT = dimT,
                nFft = nFft,
                hopLength = hopLength
            )
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to extract params for $name model")
            null
        }
    }

    private fun loadModel(assetPath: String, options: OrtSession.SessionOptions): OrtSession? {
        return try {
            val modelFile = File(context.cacheDir, assetPath.substringAfterLast("/"))
            if (!modelFile.exists()) {
                Timber.tag(TAG).i("Extracting model: $assetPath...")
                context.assets.open(assetPath).use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }
            val session = ortEnvironment!!.createSession(modelFile.absolutePath, options)
            Timber.tag(TAG).i("Loaded: $assetPath (${modelFile.length() / 1024 / 1024}MB) - $backendType")
            session
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to load $assetPath")
            null
        }
    }

    private var usingNnapi = false
    private var usingQnn = false
    private var backendType = "CPU"

    private fun createSessionOptions(): OrtSession.SessionOptions {
        android.util.Log.i(TAG, "========== createSessionOptions START ==========")
        return OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setMemoryPatternOptimization(true)

            // Set high-performance CPU threading as baseline
            val numCores = Runtime.getRuntime().availableProcessors()
            setIntraOpNumThreads(numCores)
            setInterOpNumThreads(maxOf(1, numCores / 2))

            // Enable parallel execution mode
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)

            // Get native library path for QNN libs
            val nativeLibPath = context.applicationInfo.nativeLibraryDir
            android.util.Log.i(TAG, "Native lib path: $nativeLibPath")

            // Set ADSP_LIBRARY_PATH for Hexagon skel loading
            try {
                Os.setenv("ADSP_LIBRARY_PATH", nativeLibPath, true)
                Os.setenv("LD_LIBRARY_PATH", "$nativeLibPath:${Os.getenv("LD_LIBRARY_PATH") ?: ""}", true)
                Timber.tag(TAG).i("Set ADSP_LIBRARY_PATH=$nativeLibPath")
            } catch (e: Exception) {
                Timber.tag(TAG).w("Failed to set ADSP_LIBRARY_PATH: ${e.message}")
            }

            // Try QNN HTP (NPU) with FP16 model
            var qnnSuccess = false

            // List available libraries
            android.util.Log.i(TAG, "=== QNN DEBUG INFO ===")
            android.util.Log.i(TAG, "Native lib path: $nativeLibPath")
            val qnnLibs = java.io.File(nativeLibPath).listFiles()?.filter {
                it.name.contains("Qnn") || it.name.contains("Htp")
            } ?: emptyList()
            qnnLibs.forEach {
                android.util.Log.i(TAG, "  Found: ${it.name} (${it.length() / 1024}KB)")
            }

            // Check for required libs
            val hasGpu = java.io.File("$nativeLibPath/libQnnGpu.so").exists()
            val hasHtp = java.io.File("$nativeLibPath/libQnnHtp.so").exists()
            val hasHtpPrepare = java.io.File("$nativeLibPath/libQnnHtpPrepare.so").exists()
            val hasHtpV75Stub = java.io.File("$nativeLibPath/libQnnHtpV75Stub.so").exists()
            android.util.Log.i(TAG, "  Has GPU lib: $hasGpu")
            android.util.Log.i(TAG, "  Has HTP lib: $hasHtp")
            android.util.Log.i(TAG, "  Has HTP Prepare: $hasHtpPrepare")
            android.util.Log.i(TAG, "  Has HTP v75 Stub: $hasHtpV75Stub")

            // Try 1: QNN GPU (works with FP32, uses Adreno GPU)
            if (hasGpu) {
                try {
                    android.util.Log.i(TAG, "QNN attempt 1: GPU with profiling...")
                    val gpuPath = "$nativeLibPath/libQnnGpu.so"

                    // GPU options - enable profiling to see what's happening
                    val gpuOptions = mapOf(
                        "backend_path" to gpuPath,
                        "profiling_level" to "basic"
                    )
                    android.util.Log.i(TAG, "  Options: $gpuOptions")
                    addQnn(gpuOptions)
                    usingQnn = true
                    usingNnapi = false
                    backendType = "QNN GPU"
                    qnnSuccess = true
                    android.util.Log.i(TAG, "=== QNN GPU ENABLED ===")
                } catch (e: Exception) {
                    android.util.Log.e(TAG, "QNN GPU failed: ${e.message}")
                    e.printStackTrace()
                }
            }

            // Try 2: QNN HTP (Hexagon NPU) - needs quantized models for best performance
            // but let's try anyway to see what happens
            if (!qnnSuccess && hasHtp) {
                try {
                    android.util.Log.i(TAG, "QNN attempt 2: HTP...")
                    val htpPath = "$nativeLibPath/libQnnHtp.so"

                    // HTP options
                    val htpOptions = mapOf(
                        "backend_path" to htpPath,
                        "profiling_level" to "basic"
                    )
                    android.util.Log.i(TAG, "  Options: $htpOptions")
                    addQnn(htpOptions)
                    usingQnn = true
                    usingNnapi = false
                    backendType = "QNN HTP"
                    qnnSuccess = true
                    android.util.Log.i(TAG, "=== QNN HTP ENABLED ===")
                } catch (e: Exception) {
                    android.util.Log.e(TAG, "QNN HTP failed: ${e.message}")
                    e.printStackTrace()
                }
            }

            // Fallback to NNAPI
            if (!qnnSuccess) {
                try {
                    Timber.tag(TAG).i("Trying NNAPI...")
                    val nnapiFlags = java.util.EnumSet.of(
                        ai.onnxruntime.providers.NNAPIFlags.USE_FP16
                    )
                    addNnapi(nnapiFlags)
                    usingNnapi = true
                    usingQnn = false
                    backendType = "NNAPI"
                    Timber.tag(TAG).i("=== NNAPI ENABLED (FP16) ===")
                } catch (e2: Exception) {
                    Timber.tag(TAG).e(e2, "NNAPI also failed: ${e2.message}")

                    // Fallback: Standard CPU
                    usingQnn = false
                    usingNnapi = false
                    backendType = "CPU ($numCores cores)"
                    Timber.tag(TAG).i("Using standard CPU backend")
                }
            }
        }
    }

    /**
     * Initialize FFT tables for a specific size (next power of 2 >= nFft).
     * Called once per unique FFT size used by models.
     */
    private fun initFftTablesForSize(nFft: Int) {
        // Next power of 2 >= nFft
        var fftSize = 1
        while (fftSize < nFft) fftSize *= 2

        if (fftCosTable.containsKey(fftSize)) return  // Already initialized

        val n = fftSize
        val cosTable = FloatArray(n / 2)
        val sinTable = FloatArray(n / 2)
        for (i in 0 until n / 2) {
            val angle = -2.0 * PI * i / n
            cosTable[i] = cos(angle).toFloat()
            sinTable[i] = sin(angle).toFloat()
        }
        fftCosTable[fftSize] = cosTable
        fftSinTable[fftSize] = sinTable

        val bitRev = IntArray(n)
        val bits = (kotlin.math.ln(n.toDouble()) / kotlin.math.ln(2.0)).toInt()
        for (i in 0 until n) {
            var reversed = 0
            var value = i
            for (j in 0 until bits) {
                reversed = (reversed shl 1) or (value and 1)
                value = value shr 1
            }
            bitRev[i] = reversed
        }
        bitReversalTable[fftSize] = bitRev

        Timber.tag(TAG).d("Initialized FFT tables for size $fftSize (n_fft=$nFft)")
    }

    private fun getFftSize(nFft: Int): Int {
        var fftSize = 1
        while (fftSize < nFft) fftSize *= 2
        return fftSize
    }

    fun process(input: FloatArray): FloatArray {
        if (!isInitialized.get() || input.isEmpty() || input.size < MIN_INPUT_SIZE) {
            return input
        }

        return try {
            synchronized(processLock) {
                val separated = separate(input)
                if (separated != null) remixStems(separated) else input
            }
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Process failed")
            input
        }
    }

    fun separate(input: FloatArray): StemResult? {
        if (!isInitialized.get() || input.isEmpty()) return null

        synchronized(processLock) {
            if (vocalsBuffer.size != input.size) {
                vocalsBuffer = FloatArray(input.size)
                drumsBuffer = FloatArray(input.size)
                bassBuffer = FloatArray(input.size)
                otherBuffer = FloatArray(input.size)
            }
            return separateBuffered(input)
        }
    }

    private fun separateBuffered(input: FloatArray): StemResult? {
        if (audioAccumBuffer.size < CHUNK_SAMPLES_STEREO) {
            audioAccumBuffer = FloatArray(CHUNK_SAMPLES_STEREO)
            audioAccumPos = 0
            // Start fade out when we begin buffering for the first time
            isFadingOut = true
            fadeOutPos = 0
            Timber.tag(TAG).i("Starting fade out...")
        }

        // Check completed jobs
        val iterator = inflightJobs.iterator()
        while (iterator.hasNext()) {
            val future = iterator.next()
            if (future.isDone) {
                try {
                    val result = future.get()
                    if (result != null && result.vocals.size > 1000) {
                        resultQueue.offer(result)
                        Timber.tag(TAG).d("MDX-Net done! Queue: ${resultQueue.size}, NPU: ${npuUsagePercent.toInt()}%")
                        // Update UI with inference complete status
                        updateState(
                            if (hasStartedPlayback) SeparationState.PLAYING else SeparationState.BUFFERING,
                            if (hasStartedPlayback) 0f else resultQueue.size.toFloat() / MIN_QUEUE_BEFORE_PLAYBACK,
                            "Q:${resultQueue.size}"
                        )
                        updatePipelineStats(100, resultQueue.size, forceUpdate = true)
                    }
                } catch (e: Exception) {
                    Timber.tag(TAG).e(e, "MDX-Net job failed")
                    updateState(SeparationState.EXHAUSTED, 0f, "Error")
                }
                onnxJobCount.decrementAndGet()
                iterator.remove()
            }
        }

        // Accumulate input
        val toCopy = minOf(input.size, audioAccumBuffer.size - audioAccumPos)
        if (toCopy > 0) {
            System.arraycopy(input, 0, audioAccumBuffer, audioAccumPos, toCopy)
            audioAccumPos += toCopy
        }

        // Real-time buffer progress update
        val bufferProgress = audioAccumPos.toFloat() / CHUNK_SAMPLES_STEREO
        val bufferPct = (bufferProgress * 100).toInt()
        if (!hasStartedPlayback) {
            updateState(SeparationState.BUFFERING, bufferProgress, "Q:${resultQueue.size}")
            updatePipelineStats(bufferPct, resultQueue.size)
        }

        // Log buffer progress every ~0.5 seconds
        if (audioAccumPos % (INPUT_SAMPLE_RATE / 2) < input.size) {
            val pct = (bufferProgress * 100).toInt()
            Timber.tag(TAG).i("Buffer: $pct% ($audioAccumPos/$CHUNK_SAMPLES_STEREO) | Queue: ${resultQueue.size} | Jobs: ${onnxJobCount.get()} | Playing: $hasStartedPlayback")
        }

        // Submit inference job when buffer is full - allow parallel jobs
        if (onnxJobCount.get() < MAX_PARALLEL_JOBS && audioAccumPos >= CHUNK_SAMPLES_STEREO) {
            val audio: FloatArray
            val hasOverlap = overlapBuffer.isNotEmpty()

            if (hasOverlap) {
                audio = FloatArray(overlapBuffer.size + audioAccumPos)
                System.arraycopy(overlapBuffer, 0, audio, 0, overlapBuffer.size)
                System.arraycopy(audioAccumBuffer, 0, audio, overlapBuffer.size, audioAccumPos)
            } else {
                audio = audioAccumBuffer.copyOf(audioAccumPos)
            }

            val overlapStart = maxOf(0, audioAccumPos - OVERLAP_SAMPLES * 2)
            overlapBuffer = audioAccumBuffer.copyOfRange(overlapStart, audioAccumPos)
            audioAccumPos = 0
            onnxJobCount.incrementAndGet()

            Timber.tag(TAG).i("Launching KUIELab inference (${audio.size} samples = ${audio.size / INPUT_SAMPLE_RATE / 2}s)")
            updateState(SeparationState.PROCESSING_STFT, 0f, "Q:${resultQueue.size}")

            val future = onnxExecutor.submit<StemResult?> {
                val result = runKuielabInference(audio) ?: return@submit null

                if (hasOverlap) {
                    val trimAmount = OVERLAP_SAMPLES * 2 - CROSSFADE
                    if (trimAmount > 0 && result.vocals.size > trimAmount) {
                        StemResult(
                            result.vocals.copyOfRange(trimAmount, result.vocals.size),
                            result.drums.copyOfRange(trimAmount, result.drums.size),
                            result.bass.copyOfRange(trimAmount, result.bass.size),
                            result.other.copyOfRange(trimAmount, result.other.size)
                        )
                    } else result
                } else result
            }
            inflightJobs.offer(future)
        }

        // Reset playback state if queue is empty and no current result
        // This allows re-buffering after audio stream restarts
        if (hasStartedPlayback && resultQueue.isEmpty() && currentResult == null) {
            Timber.tag(TAG).i("Queue empty, resetting to buffering mode")
            hasStartedPlayback = false
        }

        if (!hasStartedPlayback && resultQueue.size >= MIN_QUEUE_BEFORE_PLAYBACK) {
            currentResult = resultQueue.poll()
            playbackPos = 0
            hasStartedPlayback = true
            isFadingIn = true
            fadeInPos = 0
            Timber.tag(TAG).i("Starting playback with fade in!")
        }

        // Store input for passthrough fallback - reuse buffer
        if (passthroughBuffer.size < input.size) {
            passthroughBuffer = FloatArray(input.size)
        }
        System.arraycopy(input, 0, passthroughBuffer, 0, input.size)

        if (!hasStartedPlayback) {
            // State already updated above with real-time buffer progress
            // Fade out original audio then silence while buffering
            for (i in 0 until input.size) {
                val fade = if (isFadingOut && fadeOutPos < FADE_SAMPLES) {
                    val f = 1f - (fadeOutPos.toFloat() / FADE_SAMPLES)
                    fadeOutPos++
                    if (fadeOutPos >= FADE_SAMPLES) isFadingOut = false
                    f
                } else {
                    0f
                }
                // During fade out, output faded original audio
                vocalsBuffer[i] = passthroughBuffer[i] * fade * 0.25f
                drumsBuffer[i] = passthroughBuffer[i] * fade * 0.25f
                bassBuffer[i] = passthroughBuffer[i] * fade * 0.25f
                otherBuffer[i] = passthroughBuffer[i] * fade * 0.25f
            }
            return StemResult(vocalsBuffer, drumsBuffer, bassBuffer, otherBuffer)
        }

        return playbackWithCrossfade(input)
    }

    /**
     * KUIELab MDX-Net inference - COMBINED 4-stem model (single inference, 4 outputs)
     *
     * Takes stereo complex spectrogram [batch=1, dim_c=4, dim_f=2048, dim_t=256]
     * Outputs: vocals, drums, bass, other [batch=1, dim_c=4, dim_f=2048, dim_t=256] each
     */
    private fun runKuielabInference(input: FloatArray): StemResult? {
        val startTime = System.nanoTime()
        val env = ortEnvironment ?: return null
        val session = combinedSession ?: return null
        val params = combinedParams ?: return null

        try {
            // Deinterleave stereo
            val numSamplesIn = input.size / 2
            val leftChannelIn = FloatArray(numSamplesIn)
            val rightChannelIn = FloatArray(numSamplesIn)
            for (i in 0 until numSamplesIn) {
                leftChannelIn[i] = input[i * 2]
                rightChannelIn[i] = input[i * 2 + 1]
            }

            // DOWNSAMPLE to 22kHz for faster processing
            val numSamples = numSamplesIn / DOWNSAMPLE_FACTOR
            val leftChannel = downsample(leftChannelIn, numSamples)
            val rightChannel = downsample(rightChannelIn, numSamples)
            Timber.tag(TAG).d("Downsampled: $numSamplesIn -> $numSamples samples")

            // STFT phase - single STFT for combined model
            updateState(SeparationState.PROCESSING_STFT, 0f, "STFT")
            updatePipelineStats(100, resultQueue.size)
            val stftStart = System.nanoTime()

            val stftResult = computeStereoComplexStft(leftChannel, rightChannel, params)

            val stftTime = (System.nanoTime() - stftStart) / 1_000_000
            lastStftMs = stftTime.toInt()
            updatePipelineStats(100, resultQueue.size)

            // SINGLE NPU INFERENCE: Combined model produces all 4 stems
            val backendShort = if (usingQnn) "HTP" else if (usingNnapi) "NPU" else "CPU"
            updateState(SeparationState.PROCESSING_NPU, 0f, "$backendShort")
            val inferStart = System.nanoTime()

            // Estimated inference time based on last run (or default 2000ms)
            val estimatedMs = if (lastNpuMs > 0) lastNpuMs else 2000

            // Start progress updater thread - updates UI while inference runs
            val progressUpdater = Thread {
                try {
                    val startTime = System.currentTimeMillis()
                    while (!Thread.currentThread().isInterrupted) {
                        val elapsed = System.currentTimeMillis() - startTime
                        // Asymptotic progress - keeps moving but slows near 100%
                        val linearProgress = elapsed.toFloat() / estimatedMs
                        val progress = if (linearProgress < 0.9f) {
                            linearProgress
                        } else {
                            // After 90%, slow down asymptotically: 0.9 + 0.099 * (1 - e^(-x))
                            0.9f + 0.099f * (1f - kotlin.math.exp(-(linearProgress - 0.9f) * 3f).toFloat())
                        }
                        val remaining = ((estimatedMs - elapsed) / 1000f).coerceAtLeast(0f)
                        updateState(SeparationState.PROCESSING_NPU, progress.coerceIn(0f, 0.999f),
                            if (remaining > 0.1f) String.format("%.1fs", remaining) else "...")
                        Thread.sleep(100)  // Update 10x per second (reduced to prevent flicker)
                    }
                } catch (e: InterruptedException) {
                    // Expected when inference completes
                }
            }.apply { start() }

            Timber.tag(TAG).d("Running COMBINED 4-stem model on $backendType")

            // Run combined model - returns 4 outputs
            val outputs = runCombinedModel(env, session, stftResult.inputTensor, params)

            // Stop progress updater
            progressUpdater.interrupt()
            val vocalsOutput = outputs[0]
            val drumsOutput = outputs[1]
            val bassOutput = outputs[2]
            val otherOutput = outputs[3]

            lastNpuMs = ((System.nanoTime() - inferStart) / 1_000_000).toInt()
            vocalsMs = lastNpuMs / 4  // Estimate per-stem time
            drumsMs = lastNpuMs / 4
            bassMs = lastNpuMs / 4
            otherMs = lastNpuMs / 4

            val totalPct = (lastNpuMs.toFloat() / CHUNK_REALTIME_MS * 100f).toInt()

            Timber.tag(TAG).i("=== $backendType COMBINED INFERENCE ===")
            Timber.tag(TAG).i("  TOTAL: ${lastNpuMs}ms ($totalPct%) [single model, 4 outputs]")
            updatePipelineStats(100, resultQueue.size)
            updateState(SeparationState.PROCESSING_NPU, 1f, "$backendShort done")

            // Send model stats to UI
            stateListener?.onModelStats(vocalsMs, drumsMs, bassMs, otherMs, lastNpuMs, usingQnn || usingNnapi, backendType)

            // ISTFT phase - reconstruct all 4 stems in parallel (same params for all)
            updateState(SeparationState.PROCESSING_ISTFT, 0f, "ISTFT×4")
            val istftStart = System.nanoTime()

            val istftFutures = listOf(
                modelExecutor.submit<Pair<FloatArray, FloatArray>> { applyOutputAndIstft(vocalsOutput, stftResult, params) },
                modelExecutor.submit<Pair<FloatArray, FloatArray>> { applyOutputAndIstft(drumsOutput, stftResult, params) },
                modelExecutor.submit<Pair<FloatArray, FloatArray>> { applyOutputAndIstft(bassOutput, stftResult, params) },
                modelExecutor.submit<Pair<FloatArray, FloatArray>> { applyOutputAndIstft(otherOutput, stftResult, params) }
            )
            val vocalsAudio = istftFutures[0].get()
            val drumsAudio = istftFutures[1].get()
            val bassAudio = istftFutures[2].get()
            val otherAudio = istftFutures[3].get()

            val istftTime = (System.nanoTime() - istftStart) / 1_000_000
            lastIstftMs = istftTime.toInt()
            updatePipelineStats(100, resultQueue.size, forceUpdate = true)

            Timber.tag(TAG).d("ISTFT output: v=${vocalsAudio.first.size}, d=${drumsAudio.first.size}, b=${bassAudio.first.size}, o=${otherAudio.first.size}")

            // Find minimum output size across all stems
            val minSamples16k = minOf(
                vocalsAudio.first.size,
                drumsAudio.first.size,
                bassAudio.first.size,
                otherAudio.first.size,
                numSamples
            )
            val outputSamples = minSamples16k * DOWNSAMPLE_FACTOR

            // Upsample all stems back to 44.1kHz
            val vocalsL = upsample(vocalsAudio.first, minSamples16k, outputSamples)
            val vocalsR = upsample(vocalsAudio.second, minSamples16k, outputSamples)
            val drumsL = upsample(drumsAudio.first, minSamples16k, outputSamples)
            val drumsR = upsample(drumsAudio.second, minSamples16k, outputSamples)
            val bassL = upsample(bassAudio.first, minSamples16k, outputSamples)
            val bassR = upsample(bassAudio.second, minSamples16k, outputSamples)
            val otherL = upsample(otherAudio.first, minSamples16k, outputSamples)
            val otherR = upsample(otherAudio.second, minSamples16k, outputSamples)

            // Create interleaved stereo output arrays
            val vocalsOut = FloatArray(outputSamples * 2)
            val drumsOut = FloatArray(outputSamples * 2)
            val bassOut = FloatArray(outputSamples * 2)
            val otherOut = FloatArray(outputSamples * 2)

            for (i in 0 until outputSamples) {
                val idx = i * 2
                vocalsOut[idx] = vocalsL[i]
                vocalsOut[idx + 1] = vocalsR[i]
                drumsOut[idx] = drumsL[i]
                drumsOut[idx + 1] = drumsR[i]
                bassOut[idx] = bassL[i]
                bassOut[idx + 1] = bassR[i]
                otherOut[idx] = otherL[i]
                otherOut[idx + 1] = otherR[i]
            }

            val elapsed = (System.nanoTime() - startTime) / 1_000_000
            lastInferenceTimeMs = elapsed
            npuUsagePercent = (elapsed.toFloat() / CHUNK_REALTIME_MS * 100f)

            val status = when {
                npuUsagePercent < 50f -> "TURBO"
                npuUsagePercent < 80f -> "FAST"
                npuUsagePercent < 100f -> "OK"
                else -> "SLOW"
            }

            Timber.tag(TAG).i("[$status] NPU ${npuUsagePercent.toInt()}% | ${elapsed}ms (STFT:${lastStftMs}ms, NPU:${lastNpuMs}ms, ISTFT:${lastIstftMs}ms)")

            return StemResult(vocalsOut, drumsOut, bassOut, otherOut)

        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "KUIELab inference failed")
            return null
        }
    }

    /**
     * Run the combined 4-stem model and return all 4 outputs.
     * Output order: [vocals, drums, bass, other]
     */
    private fun runCombinedModel(env: OrtEnvironment, session: OrtSession, inputTensor: FloatArray, params: ModelParams): List<FloatArray> {
        val inputShape = longArrayOf(1, params.dimC.toLong(), params.dimF.toLong(), params.dimT.toLong())

        android.util.Log.i(TAG, "=== runCombinedModel START ===")
        android.util.Log.i(TAG, "  Input shape: [${inputShape.joinToString(", ")}]")
        android.util.Log.i(TAG, "  Input tensor size: ${inputTensor.size}")
        android.util.Log.i(TAG, "  Backend: $backendType")

        val tensorCreateStart = System.nanoTime()
        val onnxInput = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensor), inputShape)
        val tensorCreateMs = (System.nanoTime() - tensorCreateStart) / 1_000_000
        android.util.Log.i(TAG, "  Tensor creation: ${tensorCreateMs}ms")

        return try {
            android.util.Log.i(TAG, "  >>> Calling session.run() [$backendType] ...")
            val inferenceStart = System.nanoTime()

            val result = session.run(mapOf("input" to onnxInput))

            val inferenceMs = (System.nanoTime() - inferenceStart) / 1_000_000
            android.util.Log.i(TAG, "  <<< session.run() complete: ${inferenceMs}ms")

            // Performance classification
            val status = when {
                inferenceMs < 500 -> "TURBO (GPU/NPU working!)"
                inferenceMs < 2000 -> "FAST"
                inferenceMs < 5000 -> "OK"
                inferenceMs < 10000 -> "SLOW (likely CPU fallback)"
                else -> "VERY SLOW (CPU fallback confirmed)"
            }
            android.util.Log.i(TAG, "  Performance: $status")

            val outputStart = System.nanoTime()
            // Get all 4 outputs: vocals_output, drums_output, bass_output, other_output
            val vocalsOut = (result.get("vocals_output").get() as OnnxTensor).floatBuffer.array()
            val drumsOut = (result.get("drums_output").get() as OnnxTensor).floatBuffer.array()
            val bassOut = (result.get("bass_output").get() as OnnxTensor).floatBuffer.array()
            val otherOut = (result.get("other_output").get() as OnnxTensor).floatBuffer.array()
            val outputMs = (System.nanoTime() - outputStart) / 1_000_000
            android.util.Log.i(TAG, "  Output extraction: ${outputMs}ms")

            result.close()
            android.util.Log.i(TAG, "=== runCombinedModel END (total inference: ${inferenceMs}ms) ===")
            listOf(vocalsOut, drumsOut, bassOut, otherOut)
        } finally {
            onnxInput.close()
        }
    }

    /**
     * Holds STFT results for stereo complex input.
     * Contains the 4-channel tensor [L_real, L_imag, R_real, R_imag] and original spectrograms for ISTFT.
     */
    data class StereoStftResult(
        val inputTensor: FloatArray,    // [dim_c=4, dim_f, dim_t] for model input
        val leftReal: FloatArray,       // Left channel real part [frames, freqBins]
        val leftImag: FloatArray,       // Left channel imag part
        val rightReal: FloatArray,      // Right channel real part
        val rightImag: FloatArray,      // Right channel imag part
        val numFrames: Int,
        val freqBins: Int
    )

    /**
     * Compute stereo STFT for KUIELab model input.
     * MDX-Net models expect COMPLEX spectrogram: [L_real, L_imag, R_real, R_imag]
     * Output tensor format: [4, dim_f, dim_t]
     */
    private fun computeStereoComplexStft(
        leftChannel: FloatArray,
        rightChannel: FloatArray,
        params: ModelParams
    ): StereoStftResult {
        val nFft = params.nFft
        val hopLength = params.hopLength
        val dimF = params.dimF
        val dimT = params.dimT
        val freqBins = nFft / 2

        // Compute number of frames from audio length
        val numFrames = (leftChannel.size / hopLength) + 1

        val fftSize = getFftSize(nFft)
        val hannWindow = getHannWindow(nFft)

        // Allocate STFT outputs
        val leftReal = FloatArray(numFrames * freqBins)
        val leftImag = FloatArray(numFrames * freqBins)
        val rightReal = FloatArray(numFrames * freqBins)
        val rightImag = FloatArray(numFrames * freqBins)

        // Working buffers
        val paddedFrame = FloatArray(fftSize)
        val fftReal = FloatArray(fftSize)
        val fftImag = FloatArray(fftSize)

        // Compute STFT for both channels
        for (frame in 0 until numFrames) {
            val start = frame * hopLength

            // Left channel STFT
            paddedFrame.fill(0f)
            for (i in 0 until nFft) {
                val sampleIdx = start + i
                paddedFrame[i] = if (sampleIdx < leftChannel.size) leftChannel[sampleIdx] * hannWindow[i] else 0f
            }
            fftWithSize(paddedFrame, fftReal, fftImag, fftSize)

            for (f in 0 until freqBins) {
                val outIdx = frame * freqBins + f
                leftReal[outIdx] = fftReal[f]
                leftImag[outIdx] = fftImag[f]
            }

            // Right channel STFT
            paddedFrame.fill(0f)
            for (i in 0 until nFft) {
                val sampleIdx = start + i
                paddedFrame[i] = if (sampleIdx < rightChannel.size) rightChannel[sampleIdx] * hannWindow[i] else 0f
            }
            fftWithSize(paddedFrame, fftReal, fftImag, fftSize)

            for (f in 0 until freqBins) {
                val outIdx = frame * freqBins + f
                rightReal[outIdx] = fftReal[f]
                rightImag[outIdx] = fftImag[f]
            }
        }

        // Format as model input tensor: [dim_c=4, dim_f, dim_t]
        // MDX-Net expects COMPLEX: [L_real, L_imag, R_real, R_imag]
        val tensorSize = params.dimC * dimF * dimT
        val inputTensor = FloatArray(tensorSize)

        val frameLimit = minOf(numFrames, dimT)
        val freqLimit = minOf(freqBins, dimF)

        for (t in 0 until frameLimit) {
            for (f in 0 until freqLimit) {
                val srcIdx = t * freqBins + f
                // Tensor layout: [channel][freq][time] -> index = c*dimF*dimT + f*dimT + t

                if (srcIdx < leftReal.size) {
                    val ch0Idx = 0 * dimF * dimT + f * dimT + t  // L_real
                    val ch1Idx = 1 * dimF * dimT + f * dimT + t  // L_imag
                    val ch2Idx = 2 * dimF * dimT + f * dimT + t  // R_real
                    val ch3Idx = 3 * dimF * dimT + f * dimT + t  // R_imag

                    inputTensor[ch0Idx] = leftReal[srcIdx]
                    inputTensor[ch1Idx] = leftImag[srcIdx]
                    inputTensor[ch2Idx] = rightReal[srcIdx]
                    inputTensor[ch3Idx] = rightImag[srcIdx]
                }
            }
        }

        return StereoStftResult(inputTensor, leftReal, leftImag, rightReal, rightImag, numFrames, freqBins)
    }

    /**
     * Apply model output and do ISTFT to get stereo audio.
     * Model receives complex input [L_real, L_imag, R_real, R_imag]
     * Model outputs separated complex spectrogram in same format.
     * Returns (leftChannel, rightChannel).
     */
    private fun applyOutputAndIstft(
        output: FloatArray,
        stft: StereoStftResult,
        params: ModelParams
    ): Pair<FloatArray, FloatArray> {
        val dimF = params.dimF
        val dimT = params.dimT
        val freqBins = stft.freqBins
        val numFrames = stft.numFrames

        val frameLimit = minOf(numFrames, dimT)
        val freqLimit = minOf(freqBins, dimF)

        // Log output stats for debugging
        if (output.isNotEmpty()) {
            var minVal = Float.MAX_VALUE
            var maxVal = Float.MIN_VALUE
            var sum = 0.0
            for (v in output) {
                if (v < minVal) minVal = v
                if (v > maxVal) maxVal = v
                sum += v
            }
            val avg = sum / output.size
            Timber.tag(TAG).d("Output stats: min=$minVal, max=$maxVal, avg=$avg, size=${output.size}")
        }

        // Initialize output spectrogram arrays
        val outLeftReal = FloatArray(numFrames * freqBins)
        val outLeftImag = FloatArray(numFrames * freqBins)
        val outRightReal = FloatArray(numFrames * freqBins)
        val outRightImag = FloatArray(numFrames * freqBins)

        // Model outputs separated COMPLEX spectrogram directly
        // Output format: [L_real, L_imag, R_real, R_imag]
        for (t in 0 until frameLimit) {
            for (f in 0 until freqLimit) {
                val srcIdx = t * freqBins + f
                val ch0Idx = 0 * dimF * dimT + f * dimT + t  // L_real
                val ch1Idx = 1 * dimF * dimT + f * dimT + t  // L_imag
                val ch2Idx = 2 * dimF * dimT + f * dimT + t  // R_real
                val ch3Idx = 3 * dimF * dimT + f * dimT + t  // R_imag

                if (ch3Idx < output.size && srcIdx < outLeftReal.size) {
                    // Direct complex output from model
                    outLeftReal[srcIdx] = output[ch0Idx]
                    outLeftImag[srcIdx] = output[ch1Idx]
                    outRightReal[srcIdx] = output[ch2Idx]
                    outRightImag[srcIdx] = output[ch3Idx]
                }
            }
        }

        // ISTFT for both channels
        val leftAudio = computeIstftWithParams(outLeftReal, outLeftImag, numFrames, params)
        val rightAudio = computeIstftWithParams(outRightReal, outRightImag, numFrames, params)

        return Pair(leftAudio, rightAudio)
    }

    /**
     * Compute inverse STFT with model-specific parameters.
     */
    private fun computeIstftWithParams(specReal: FloatArray, specImag: FloatArray, numFrames: Int, params: ModelParams): FloatArray {
        val nFft = params.nFft
        val hopLength = params.hopLength
        val freqBins = nFft / 2
        val fftSize = getFftSize(nFft)
        val hannWindow = getHannWindow(nFft)

        val outputLength = numFrames * hopLength + nFft
        val output = FloatArray(outputLength)
        val windowSum = FloatArray(outputLength)

        val fftReal = FloatArray(fftSize)
        val fftImag = FloatArray(fftSize)
        val ifftReal = FloatArray(fftSize)

        for (frame in 0 until numFrames) {
            // Prepare frequency domain data
            fftReal.fill(0f)
            fftImag.fill(0f)

            for (f in 0 until freqBins) {
                val idx = frame * freqBins + f
                if (idx < specReal.size) {
                    fftReal[f] = specReal[idx]
                    fftImag[f] = specImag[idx]
                    // Conjugate symmetry for real signal
                    if (f > 0 && f < freqBins) {
                        fftReal[fftSize - f] = specReal[idx]
                        fftImag[fftSize - f] = -specImag[idx]
                    }
                }
            }

            // Inverse FFT
            ifftWithSize(fftReal, fftImag, ifftReal, fftSize)

            // Overlap-add with window
            val start = frame * hopLength
            for (i in 0 until nFft) {
                if (start + i < output.size) {
                    output[start + i] += ifftReal[i] * hannWindow[i]
                    windowSum[start + i] += hannWindow[i] * hannWindow[i]
                }
            }
        }

        // Normalize by window sum
        for (i in output.indices) {
            if (windowSum[i] > 1e-8f) {
                output[i] /= windowSum[i]
            }
        }

        return output
    }

    /**
     * Simple downsampling by factor of 2 with averaging filter.
     * Fast and good enough for source separation.
     */
    private fun downsample(input: FloatArray, outputSize: Int): FloatArray {
        val output = FloatArray(outputSize)
        for (i in 0 until outputSize) {
            val idx = i * DOWNSAMPLE_FACTOR
            // Average filter for anti-aliasing
            var sum = 0f
            for (j in 0 until DOWNSAMPLE_FACTOR) {
                if (idx + j < input.size) sum += input[idx + j]
            }
            output[i] = sum / DOWNSAMPLE_FACTOR
        }
        return output
    }

    /**
     * Simple upsampling with linear interpolation.
     * Fast and smooth enough for audio.
     */
    private fun upsample(input: FloatArray, inputSize: Int, outputSize: Int): FloatArray {
        val output = FloatArray(outputSize)
        val ratio = inputSize.toFloat() / outputSize
        for (i in 0 until outputSize) {
            val srcPos = i * ratio
            val srcIdx = srcPos.toInt()
            val frac = srcPos - srcIdx
            val s0 = if (srcIdx < inputSize) input[srcIdx] else 0f
            val s1 = if (srcIdx + 1 < inputSize) input[srcIdx + 1] else s0
            output[i] = s0 + (s1 - s0) * frac
        }
        return output
    }

    /**
     * FFT with specific size - uses precomputed tables for that size.
     */
    private fun fftWithSize(input: FloatArray, outReal: FloatArray, outImag: FloatArray, fftSize: Int) {
        val n = fftSize
        val bitRev = bitReversalTable[fftSize] ?: return
        val cosTable = fftCosTable[fftSize] ?: return
        val sinTable = fftSinTable[fftSize] ?: return

        for (i in 0 until n) {
            outReal[bitRev[i]] = if (i < input.size) input[i] else 0f
            outImag[bitRev[i]] = 0f
        }

        var size = 2
        while (size <= n) {
            val halfSize = size / 2
            val step = n / size
            for (i in 0 until n step size) {
                for (j in 0 until halfSize) {
                    val k = j * step
                    val tReal = cosTable[k] * outReal[i + j + halfSize] - sinTable[k] * outImag[i + j + halfSize]
                    val tImag = sinTable[k] * outReal[i + j + halfSize] + cosTable[k] * outImag[i + j + halfSize]
                    outReal[i + j + halfSize] = outReal[i + j] - tReal
                    outImag[i + j + halfSize] = outImag[i + j] - tImag
                    outReal[i + j] += tReal
                    outImag[i + j] += tImag
                }
            }
            size *= 2
        }
    }

    /**
     * Inverse FFT with specific size - uses precomputed tables for that size.
     */
    private fun ifftWithSize(real: FloatArray, imag: FloatArray, outReal: FloatArray, fftSize: Int) {
        val n = fftSize
        val bitRev = bitReversalTable[fftSize] ?: return
        val cosTable = fftCosTable[fftSize] ?: return
        val sinTable = fftSinTable[fftSize] ?: return

        // Conjugate input
        val conjImag = FloatArray(n) { -imag[it] }

        // Forward FFT on conjugated signal
        val tempReal = FloatArray(n)
        val tempImag = FloatArray(n)

        for (i in 0 until n) {
            tempReal[bitRev[i]] = real[i]
            tempImag[bitRev[i]] = conjImag[i]
        }

        var size = 2
        while (size <= n) {
            val halfSize = size / 2
            val step = n / size
            for (i in 0 until n step size) {
                for (j in 0 until halfSize) {
                    val k = j * step
                    val tReal = cosTable[k] * tempReal[i + j + halfSize] - sinTable[k] * tempImag[i + j + halfSize]
                    val tImag = sinTable[k] * tempReal[i + j + halfSize] + cosTable[k] * tempImag[i + j + halfSize]
                    tempReal[i + j + halfSize] = tempReal[i + j] - tReal
                    tempImag[i + j + halfSize] = tempImag[i + j] - tImag
                    tempReal[i + j] += tReal
                    tempImag[i + j] += tImag
                }
            }
            size *= 2
        }

        // Conjugate and scale
        for (i in 0 until n) {
            outReal[i] = tempReal[i] / n
        }
    }

    private fun playbackWithCrossfade(input: FloatArray): StemResult {
        val current = currentResult ?: run {
            // No current result - use last played chunk for seamless fallback
            val fallback = lastPlayedResult
            if (fallback != null && fallback.vocals.isNotEmpty()) {
                updateState(SeparationState.EXHAUSTED, 0f, "Looping...")
                // Loop from the last chunk to avoid silence
                val loopSize = minOf(input.size, fallback.vocals.size)
                for (i in 0 until loopSize) {
                    val idx = i % fallback.vocals.size
                    vocalsBuffer[i] = fallback.vocals[idx]
                    drumsBuffer[i] = fallback.drums[idx]
                    bassBuffer[i] = fallback.bass[idx]
                    otherBuffer[i] = fallback.other[idx]
                }
                return StemResult(vocalsBuffer, drumsBuffer, bassBuffer, otherBuffer)
            }
            // No fallback available yet (cold start) - silence
            updateState(SeparationState.EXHAUSTED, 0f, "Waiting...")
            for (i in 0 until input.size) {
                vocalsBuffer[i] = 0f
                drumsBuffer[i] = 0f
                bassBuffer[i] = 0f
                otherBuffer[i] = 0f
            }
            return StemResult(vocalsBuffer, drumsBuffer, bassBuffer, otherBuffer)
        }

        val effectiveEnd = maxOf(0, current.vocals.size - CROSSFADE)
        val remaining = maxOf(0, effectiveEnd - playbackPos)

        if (remaining >= input.size) {
            System.arraycopy(current.vocals, playbackPos, vocalsBuffer, 0, input.size)
            System.arraycopy(current.drums, playbackPos, drumsBuffer, 0, input.size)
            System.arraycopy(current.bass, playbackPos, bassBuffer, 0, input.size)
            System.arraycopy(current.other, playbackPos, otherBuffer, 0, input.size)

            // Apply fade in if starting playback
            if (isFadingIn) {
                for (i in 0 until input.size) {
                    if (fadeInPos < FADE_SAMPLES) {
                        val fade = fadeInPos.toFloat() / FADE_SAMPLES
                        vocalsBuffer[i] *= fade
                        drumsBuffer[i] *= fade
                        bassBuffer[i] *= fade
                        otherBuffer[i] *= fade
                        fadeInPos++
                    } else {
                        isFadingIn = false
                        break
                    }
                }
            }

            playbackPos += input.size
            lastPlayedResult = current  // Save for seamless fallback
            updateState(SeparationState.PLAYING, playbackPos.toFloat() / current.vocals.size, "Q:${resultQueue.size}")
        } else {
            val next = resultQueue.poll()
            if (next != null) {
                var outPos = 0
                if (remaining > 0) {
                    System.arraycopy(current.vocals, playbackPos, vocalsBuffer, 0, remaining)
                    System.arraycopy(current.drums, playbackPos, drumsBuffer, 0, remaining)
                    System.arraycopy(current.bass, playbackPos, bassBuffer, 0, remaining)
                    System.arraycopy(current.other, playbackPos, otherBuffer, 0, remaining)
                    outPos = remaining
                }

                val crossfadeStart = effectiveEnd
                val crossfadeSamples = minOf(CROSSFADE, input.size - outPos, next.vocals.size)
                if (crossfadeSamples > 0) {
                    for (i in 0 until crossfadeSamples) {
                        val fadeOut = 1f - crossfadeTable[i]
                        val fadeIn = crossfadeTable[i]
                        vocalsBuffer[outPos + i] = current.vocals[crossfadeStart + i] * fadeOut + next.vocals[i] * fadeIn
                        drumsBuffer[outPos + i] = current.drums[crossfadeStart + i] * fadeOut + next.drums[i] * fadeIn
                        bassBuffer[outPos + i] = current.bass[crossfadeStart + i] * fadeOut + next.bass[i] * fadeIn
                        otherBuffer[outPos + i] = current.other[crossfadeStart + i] * fadeOut + next.other[i] * fadeIn
                    }
                    outPos += crossfadeSamples
                }

                var nextPlaybackPos = crossfadeSamples
                val remainingOutput = input.size - outPos
                if (remainingOutput > 0) {
                    val toCopy = minOf(remainingOutput, next.vocals.size - nextPlaybackPos)
                    System.arraycopy(next.vocals, nextPlaybackPos, vocalsBuffer, outPos, toCopy)
                    System.arraycopy(next.drums, nextPlaybackPos, drumsBuffer, outPos, toCopy)
                    System.arraycopy(next.bass, nextPlaybackPos, bassBuffer, outPos, toCopy)
                    System.arraycopy(next.other, nextPlaybackPos, otherBuffer, outPos, toCopy)
                    nextPlaybackPos += toCopy
                }

                currentResult = next
                lastPlayedResult = next  // Save for seamless fallback
                playbackPos = nextPlaybackPos
                updateState(SeparationState.PLAYING, 0f, "Q:${resultQueue.size}")
            } else {
                // Queue empty - use last chunk for seamless continuation
                Timber.tag(TAG).w("Queue empty - using fallback!")

                val fallback = lastPlayedResult ?: current
                if (fallback.vocals.isNotEmpty()) {
                    updateState(SeparationState.EXHAUSTED, 0f, "Looping...")
                    // Continue from last played chunk (loop it)
                    val loopSize = minOf(input.size, fallback.vocals.size)
                    for (i in 0 until loopSize) {
                        vocalsBuffer[i] = fallback.vocals[i % fallback.vocals.size]
                        drumsBuffer[i] = fallback.drums[i % fallback.drums.size]
                        bassBuffer[i] = fallback.bass[i % fallback.bass.size]
                        otherBuffer[i] = fallback.other[i % fallback.other.size]
                    }
                } else {
                    // No fallback - silence (shouldn't happen after cold start)
                    updateState(SeparationState.EXHAUSTED, 0f, "Waiting...")
                    for (i in 0 until input.size) {
                        vocalsBuffer[i] = 0f
                        drumsBuffer[i] = 0f
                        bassBuffer[i] = 0f
                        otherBuffer[i] = 0f
                    }
                }

                // Mark currentResult as consumed so we can re-buffer
                currentResult = null
            }
        }

        return StemResult(vocalsBuffer, drumsBuffer, bassBuffer, otherBuffer)
    }

    private fun remixStems(stems: StemResult): FloatArray {
        val size = stems.vocals.size
        val output = FloatArray(size)
        val vLevel = vocalsLevel.get()
        val dLevel = drumsLevel.get()
        val bLevel = bassLevel.get()
        val oLevel = otherLevel.get()

        for (i in 0 until size) {
            output[i] = (stems.vocals[i] * vLevel + stems.drums[i] * dLevel +
                        stems.bass[i] * bLevel + stems.other[i] * oLevel).coerceIn(-1f, 1f)
        }
        return output
    }

    // Level controls
    fun setVocalsLevel(level: Float) = vocalsLevel.set(level.coerceIn(0f, 2f))
    fun setDrumsLevel(level: Float) = drumsLevel.set(level.coerceIn(0f, 2f))
    fun setBassLevel(level: Float) = bassLevel.set(level.coerceIn(0f, 2f))
    fun setOtherLevel(level: Float) = otherLevel.set(level.coerceIn(0f, 2f))

    fun getVocalsLevel() = vocalsLevel.get()
    fun getDrumsLevel() = drumsLevel.get()
    fun getBassLevel() = bassLevel.get()
    fun getOtherLevel() = otherLevel.get()

    fun setKaraokeMode() { setVocalsLevel(0f); setDrumsLevel(1f); setBassLevel(1f); setOtherLevel(1f) }
    fun setVocalsOnlyMode() { setVocalsLevel(1f); setDrumsLevel(0f); setBassLevel(0f); setOtherLevel(0f) }
    fun resetLevels() { setVocalsLevel(1f); setDrumsLevel(1f); setBassLevel(1f); setOtherLevel(1f) }

    fun release() {
        if (!isInitialized.getAndSet(false)) return
        synchronized(processLock) {
            inflightJobs.forEach { it.cancel(true) }
            inflightJobs.clear()
            onnxJobCount.set(0)
            onnxExecutor.shutdownNow()
            modelExecutor.shutdownNow()

            // Close combined model session
            combinedSession?.close()
            combinedSession = null
            combinedParams = null
            ortEnvironment?.close()
            ortEnvironment = null

            resultQueue.clear()
            currentResult = null
            hasStartedPlayback = false
        }
    }

    fun isAvailable() = isInitialized.get()
    fun isUsing4Stem() = true  // Full 4-stem parallel mode
    fun isUsingOnnx() = isInitialized.get()
    fun isUsingQnn() = usingQnn
    fun getBackendName(): String {
        if (!isInitialized.get()) return "None"
        return "KUIELab $backendType"
    }

    fun resetFilters() {
        synchronized(processLock) {
            audioAccumPos = 0
            resultQueue.clear()
            currentResult = null
            lastPlayedResult = null
            playbackPos = 0
            hasStartedPlayback = false
            overlapBuffer = FloatArray(0)
            // Reset fade states
            isFadingOut = false
            isFadingIn = false
            fadeOutPos = 0
            fadeInPos = 0
            // passthroughBuffer is reused, no need to reset
            inflightJobs.forEach { it.cancel(true) }
            inflightJobs.clear()
            onnxJobCount.set(0)
        }
    }

    fun isModelAvailable(): Boolean {
        return try {
            context.assets.open(MODEL_COMBINED).use { true }
        } catch (e: Exception) { false }
    }

    data class StemResult(
        val vocals: FloatArray,
        val drums: FloatArray,
        val bass: FloatArray,
        val other: FloatArray
    )
}
