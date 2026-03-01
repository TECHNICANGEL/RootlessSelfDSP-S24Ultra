package me.timschneeberger.rootlessjamesdsp.view

import android.app.ActivityManager
import android.content.Context
import android.graphics.*
import android.os.Debug
import android.util.AttributeSet
import android.view.View
import java.io.RandomAccessFile
import kotlin.math.roundToInt

/**
 * S24 ULTRA - Technical Stats Monitor
 *
 * Real-time display of:
 * - CPU usage per core (X4, A720, A520)
 * - RAM usage (total app allocation)
 * - Processing latency (ms)
 * - Sample rate
 * - Buffer fill percentage
 * - Thermal (SoC temperature)
 * - Throughput (MB/s)
 *
 * Updates at 10Hz (every 100ms) - only when receiving data
 */
class StatsMonitorView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Note: Removed forced LAYER_TYPE_HARDWARE - it causes VulkanSurface errors
    // during surface transitions. Default LAYER_TYPE_NONE lets the system decide
    // when to use hardware acceleration, which is already enabled for most operations.

    // Stats data
    private var cpuUsage = 0f
    private var ramUsageMB = 0L
    private var latencyMs = 0f
    private var sampleRateKhz = 48
    private var bufferFillPercent = 0f
    private var thermalC = 0f
    private var throughputMBps = 0f

    // Animation state - RAW POWER: Always animate
    private var isAnimating = false
    private var lastDataUpdate = 0L
    private val animationTimeout = 2000L // RAW POWER: Keep animating longer

    // PERFORMANCE: Cache expensive I/O reads to avoid blocking the UI thread
    private var cachedCpuUsage = 0f
    private var cachedThermal = 0f
    private var lastCpuReadTime = 0L
    private var lastThermalReadTime = 0L
    private val cpuReadIntervalMs = 500L   // Read CPU every 500ms max
    private val thermalReadIntervalMs = 2000L  // Read thermal every 2 seconds max
    private var previousCpuIdle = 0L
    private var previousCpuTotal = 0L
    // PERFORMANCE: Cache which thermal path works to avoid repeated file open attempts
    private var workingThermalPath: String? = null

    // CPU per-core (Snapdragon 8 Gen 3) - RAW POWER: All cores tracked
    private val cpuCores = FloatArray(8) // 1xX4 + 5xA720 + 2xA520
    private val coreLabels = listOf("X4", "A720", "A720", "A720", "A720", "A720", "A520", "A520")

    // History graphs - RAW POWER: 120 samples = 1 second @ 120Hz
    private val cpuHistory = ArrayDeque<Float>(120)
    private val ramHistory = ArrayDeque<Long>(120)
    private val latencyHistory = ArrayDeque<Float>(120)
    private val throughputHistory = ArrayDeque<Float>(120)

    // Pre-allocated arrays for graph rendering (avoids GC on every frame)
    private val latencyGraphData = FloatArray(120)
    private val throughputGraphData = FloatArray(120)

    // PERFORMANCE: Pre-cached colors to avoid Color.parseColor() in onDraw
    companion object {
        private val COLOR_CYAN = Color.parseColor("#00E5FF")
        private val COLOR_GREEN = Color.parseColor("#00FF88")
        private val COLOR_YELLOW = Color.parseColor("#FFEB3B")
        private val COLOR_ORANGE = Color.parseColor("#FF9800")
        private val COLOR_RED = Color.parseColor("#F44336")
    }

    // PERFORMANCE: Cached formatted strings to avoid allocations in onDraw
    private var cachedCpuText = ""
    private var cachedLatencyText = ""
    private var cachedHealthText = ""
    private var cachedThroughputText = ""
    private var cachedRamText = ""
    private var cachedThermalText = ""
    private var cachedSampleRateText = ""
    // Track last values to know when to regenerate strings
    private var lastCpuValue = -1f
    private var lastLatencyValue = -1f
    private var lastHealthValue = -1f
    private var lastThroughputValue = -1f
    private var lastRamValue = -1L
    private var lastThermalValue = -1f
    private var lastSampleRate = -1

    // Paint objects
    private val bgPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#000000")
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 32f
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#808080")
        textSize = 24f
        typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
    }

    private val graphPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.parseColor("#9C27B0")
    }

    private val gridPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 1f
        color = Color.parseColor("#1A1A1A")
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Skip drawing if view is detached or surface invalid - prevents HWUI surface errors
        // Multiple checks needed because surface can be invalid even when view is attached
        if (!isAttachedToWindow || windowToken == null || display == null) {
            isAnimating = false
            return
        }

        val width = width.toFloat()
        val height = height.toFloat()

        // Skip drawing if view is too small or has invalid dimensions
        if (width < 100f || height < 100f) {
            isAnimating = false
            return
        }

        // Background
        canvas.drawColor(Color.BLACK)

        // Layout
        val col1X = 40f
        val col2X = width / 2f + 20f
        var currentY = 60f
        val lineHeight = 80f

        // === COLUMN 1 ===
        // PERFORMANCE: Update cached strings only when values change

        // CPU Usage
        if (cpuUsage != lastCpuValue) {
            cachedCpuText = "${"%.1f".format(cpuUsage)}%"
            lastCpuValue = cpuUsage
        }
        drawStat(canvas, col1X, currentY, "CPU", cachedCpuText,
            getColorForPercentage(cpuUsage))
        currentY += lineHeight

        // RAM Usage (protect against division by zero)
        val totalRam = getTotalRamMB()
        if (ramUsageMB != lastRamValue) {
            cachedRamText = "${ramUsageMB}MB / ${totalRam}MB"
            lastRamValue = ramUsageMB
        }
        drawStat(canvas, col1X, currentY, "RAM", cachedRamText,
            getColorForRam(ramUsageMB))
        currentY += lineHeight

        // Latency
        if (latencyMs != lastLatencyValue) {
            cachedLatencyText = "${"%.2f".format(latencyMs)}ms"
            lastLatencyValue = latencyMs
        }
        drawStat(canvas, col1X, currentY, "LATENCY", cachedLatencyText,
            getColorForLatency(latencyMs))
        currentY += lineHeight

        // Thermal
        if (thermalC != lastThermalValue) {
            cachedThermalText = "${thermalC.roundToInt()}°C"
            lastThermalValue = thermalC
        }
        drawStat(canvas, col1X, currentY, "THERMAL", cachedThermalText,
            getColorForThermal(thermalC))

        // === COLUMN 2 ===
        currentY = 60f

        // Sample Rate
        if (sampleRateKhz != lastSampleRate) {
            cachedSampleRateText = "${sampleRateKhz}kHz"
            lastSampleRate = sampleRateKhz
        }
        drawStat(canvas, col2X, currentY, "SAMPLE RATE", cachedSampleRateText, COLOR_CYAN)
        currentY += lineHeight

        // Buffer Health (pipeline stability)
        if (bufferFillPercent != lastHealthValue) {
            cachedHealthText = "${"%.0f".format(bufferFillPercent)}%"
            lastHealthValue = bufferFillPercent
        }
        drawStat(canvas, col2X, currentY, "HEALTH", cachedHealthText,
            getColorForHealth(bufferFillPercent))
        currentY += lineHeight

        // Throughput
        if (throughputMBps != lastThroughputValue) {
            cachedThroughputText = "${"%.1f".format(throughputMBps)}MB/s"
            lastThroughputValue = throughputMBps
        }
        drawStat(canvas, col2X, currentY, "THROUGHPUT", cachedThroughputText, COLOR_GREEN)

        // Draw mini history graphs
        val graphY = lineHeight * 4 + 40f
        val graphHeight = 80f
        val graphWidth = ((width - 80f) / 2f - 20f).coerceAtLeast(50f)

        // Only draw graphs if we have space
        if (graphY + graphHeight < height - 50f) {
            // Copy history to pre-allocated arrays (avoids GC on every frame)
            copyDequeToArray(latencyHistory, latencyGraphData)
            copyDequeToArray(throughputHistory, throughputGraphData)

            // Latency history graph (PERFORMANCE: use cached color)
            drawMiniGraph(canvas, col1X, graphY, graphWidth, graphHeight,
                latencyGraphData, latencyHistory.size, "LATENCY HISTORY", COLOR_CYAN, 50f)

            // Throughput history graph (PERFORMANCE: use cached color)
            drawMiniGraph(canvas, col2X, graphY, graphWidth, graphHeight,
                throughputGraphData, throughputHistory.size, "THROUGHPUT HISTORY", COLOR_GREEN, 100f)
        }

        // S24 Ultra badge
        textPaint.textSize = 20f
        textPaint.color = Color.parseColor("#9C27B0")
        textPaint.textAlign = Paint.Align.CENTER
        canvas.drawText("S24 ULTRA EDITION", width / 2f, height - 20f, textPaint)

        // Only continue animation if we're receiving data and view is attached with valid surface
        // This prevents idle CPU usage when no audio is playing
        // and HWUI errors when rendering to destroyed surfaces
        val animCurrentTime = System.currentTimeMillis()
        if (isAnimating && animCurrentTime - lastDataUpdate < animationTimeout && isAttachedToWindow && windowToken != null) {
            postInvalidateOnAnimation()
        } else {
            isAnimating = false
        }
    }

    override fun onDetachedFromWindow() {
        isAnimating = false
        super.onDetachedFromWindow()
    }

    override fun onVisibilityChanged(changedView: View, visibility: Int) {
        super.onVisibilityChanged(changedView, visibility)
        if (visibility != VISIBLE) {
            isAnimating = false
        }
    }

    override fun onWindowVisibilityChanged(visibility: Int) {
        super.onWindowVisibilityChanged(visibility)
        if (visibility != VISIBLE) {
            isAnimating = false
        }
    }

    private fun drawStat(canvas: Canvas, x: Float, y: Float, label: String, value: String, color: Int) {
        // Label
        labelPaint.textAlign = Paint.Align.LEFT
        canvas.drawText(label, x, y, labelPaint)

        // Value
        textPaint.textAlign = Paint.Align.LEFT
        textPaint.color = color
        textPaint.textSize = 40f
        canvas.drawText(value, x, y + 40f, textPaint)
    }

    private fun getColorForPercentage(percent: Float): Int = when {
        percent > 80f -> Color.RED
        percent > 60f -> Color.parseColor("#FF8800")
        percent > 40f -> Color.YELLOW
        else -> Color.GREEN
    }

    private fun getColorForRam(ramMB: Long): Int = when {
        ramMB > 1800 -> Color.parseColor("#9C27B0") // Purple - using 2GB as designed!
        ramMB > 1200 -> Color.parseColor("#00E5FF") // Cyan - high but ok
        ramMB > 600 -> Color.GREEN // Normal
        else -> Color.YELLOW // Low (unexpected)
    }

    private fun getColorForLatency(ms: Float): Int = when {
        ms > 10f -> Color.RED
        ms > 5f -> Color.YELLOW
        else -> Color.GREEN
    }

    private fun getColorForThermal(temp: Float): Int = when {
        temp > 50f -> Color.RED
        temp > 45f -> Color.parseColor("#FF8800")
        temp > 40f -> Color.YELLOW
        else -> Color.GREEN
    }

    private fun getColorForHealth(percent: Float): Int = when {
        percent >= 90f -> Color.GREEN
        percent >= 70f -> Color.parseColor("#00E5FF") // Cyan
        percent >= 50f -> Color.YELLOW
        percent >= 30f -> Color.parseColor("#FF8800") // Orange
        else -> Color.RED
    }

    /**
     * Copy ArrayDeque contents to pre-allocated array (avoids allocation on every frame)
     */
    private fun copyDequeToArray(deque: ArrayDeque<Float>, array: FloatArray) {
        var i = 0
        for (value in deque) {
            if (i >= array.size) break
            array[i++] = value
        }
        // Zero out rest of array
        while (i < array.size) {
            array[i++] = 0f
        }
    }

    private fun drawMiniGraph(
        canvas: Canvas,
        x: Float,
        y: Float,
        width: Float,
        height: Float,
        data: FloatArray,
        dataSize: Int,  // Actual number of valid elements in data array
        label: String,
        color: Int,
        maxValue: Float
    ) {
        // Edge cases: empty data, invalid dimensions, division by zero
        if (dataSize <= 0 || width <= 0f || height <= 0f || maxValue <= 0f) return

        // Draw label
        labelPaint.textAlign = Paint.Align.LEFT
        labelPaint.textSize = 18f
        canvas.drawText(label, x, y - 8f, labelPaint)

        // Draw background
        graphPaint.style = Paint.Style.FILL
        graphPaint.color = Color.parseColor("#0D0D0D")
        canvas.drawRoundRect(x, y, x + width, y + height, 8f, 8f, graphPaint)

        // Draw grid lines
        graphPaint.style = Paint.Style.STROKE
        graphPaint.color = Color.parseColor("#1A1A1A")
        for (i in 1..3) {
            val lineY = y + height * i / 4f
            canvas.drawLine(x, lineY, x + width, lineY, graphPaint)
        }

        // Draw data line
        if (dataSize > 1) {
            graphPaint.color = color
            graphPaint.strokeWidth = 2f
            graphPaint.style = Paint.Style.STROKE

            val stepX = width / (dataSize - 1).coerceAtLeast(1)
            var prevX = x
            var prevY = y + height - (data[0] / maxValue).coerceIn(0f, 1f) * height

            for (i in 1 until dataSize) {
                val currX = x + i * stepX
                val normalized = (data[i] / maxValue).coerceIn(0f, 1f)
                val currY = y + height - normalized * height

                canvas.drawLine(prevX, prevY, currX, currY, graphPaint)
                prevX = currX
                prevY = currY
            }

            // Draw current value dot
            graphPaint.style = Paint.Style.FILL
            canvas.drawCircle(prevX, prevY, 4f, graphPaint)
        }

        // Reset stroke width
        graphPaint.strokeWidth = 1f
    }


    private fun updateStats() {
        // CPU
        cpuUsage = getCpuUsage()

        // RAM
        val memInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memInfo)
        ramUsageMB = memInfo.totalPss / 1024L // Convert KB to MB

        // Add to history - RAW POWER: 120 samples
        cpuHistory.addLast(cpuUsage)
        ramHistory.addLast(ramUsageMB)
        if (cpuHistory.size > 120) cpuHistory.removeFirst()
        if (ramHistory.size > 120) ramHistory.removeFirst()

        // Thermal - read from SoC
        thermalC = getSocTemperature()
    }

    private fun getCpuUsage(): Float {
        // PERFORMANCE: Return cached value if within interval
        val now = System.currentTimeMillis()
        if (now - lastCpuReadTime < cpuReadIntervalMs) {
            return cachedCpuUsage
        }

        cachedCpuUsage = try {
            RandomAccessFile("/proc/stat", "r").use { reader ->
                val load = reader.readLine() ?: return@use cachedCpuUsage

                val toks = load.split(" +".toRegex())
                if (toks.size < 8) return@use cachedCpuUsage

                val idle = toks[4].toLongOrNull() ?: 0L
                val total = toks.slice(1..7).sumOf { it.toLongOrNull() ?: 0L }

                // Protect against division by zero and calculate delta from previous reading
                val deltaTotal = total - previousCpuTotal
                val deltaIdle = idle - previousCpuIdle

                previousCpuTotal = total
                previousCpuIdle = idle

                if (deltaTotal <= 0L) return@use cachedCpuUsage

                // Calculate actual CPU usage from deltas (more accurate than absolute values)
                ((deltaTotal - deltaIdle).toFloat() / deltaTotal.toFloat()) * 100f
            }
        } catch (e: Exception) {
            cachedCpuUsage
        }

        lastCpuReadTime = now
        return cachedCpuUsage
    }

    private fun getTotalRamMB(): Long {
        return try {
            val memInfo = ActivityManager.MemoryInfo()
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            activityManager?.getMemoryInfo(memInfo)
            memInfo.totalMem / (1024 * 1024)
        } catch (e: Exception) {
            0L // Return 0 if unable to get RAM info
        }
    }

    private fun getSocTemperature(): Float {
        // PERFORMANCE: Return cached value if within interval (thermal changes slowly)
        val now = System.currentTimeMillis()
        if (now - lastThermalReadTime < thermalReadIntervalMs) {
            return cachedThermal
        }

        cachedThermal = try {
            // PERFORMANCE: Use cached working path if available (avoids repeated file open failures)
            workingThermalPath?.let { path ->
                try {
                    RandomAccessFile(path, "r").use { it.readLine().toFloat() / 1000f }
                } catch (e: Exception) {
                    workingThermalPath = null  // Path no longer works, try others
                    null
                }
            } ?: run {
                // Try multiple thermal zones for Snapdragon 8 Gen 3
                val thermalPaths = listOf(
                    "/sys/class/thermal/thermal_zone0/temp",
                    "/sys/class/thermal/thermal_zone1/temp",
                    "/sys/devices/virtual/thermal/thermal_zone0/temp"
                )

                thermalPaths.firstNotNullOfOrNull { path ->
                    try {
                        val temp = RandomAccessFile(path, "r").use { it.readLine().toFloat() / 1000f }
                        workingThermalPath = path  // Cache the working path
                        temp
                    } catch (e: Exception) { null }
                } ?: cachedThermal
            }
        } catch (e: Exception) {
            cachedThermal
        }

        lastThermalReadTime = now
        return cachedThermal
    }

    /**
     * Update from service
     */
    fun updateFromService(latency: Float, sampleRate: Int, bufferFill: Float, throughput: Float) {
        // Update animation state
        lastDataUpdate = System.currentTimeMillis()
        val wasAnimating = isAnimating
        isAnimating = true

        // Sanitize inputs - handle NaN and Infinity
        this.latencyMs = latency.sanitize(0f).coerceIn(0f, 1000f)
        this.sampleRateKhz = (sampleRate / 1000).coerceIn(0, 384)
        this.bufferFillPercent = (bufferFill.sanitize(0f) * 100f).coerceIn(0f, 100f)
        this.throughputMBps = throughput.sanitize(0f).coerceIn(0f, 10000f)

        // Add to history for graphs (sanitized values) - RAW POWER: 120 samples
        latencyHistory.addLast(this.latencyMs)
        throughputHistory.addLast(this.throughputMBps)
        if (latencyHistory.size > 120) latencyHistory.removeFirst()
        if (throughputHistory.size > 120) throughputHistory.removeFirst()

        // Update internal stats when receiving service data
        updateStats()

        // Only call invalidate if we weren't already animating
        if (!wasAnimating && isAttachedToWindow && windowToken != null) {
            invalidate()
        }
    }

    /**
     * Sanitize float value - replace NaN/Infinity with default
     */
    private fun Float.sanitize(default: Float): Float {
        return if (this.isNaN() || this.isInfinite()) default else this
    }

    /**
     * Reset stats to idle state
     */
    fun reset() {
        isAnimating = false
        cpuUsage = 0f
        ramUsageMB = 0L
        latencyMs = 0f
        bufferFillPercent = 0f
        throughputMBps = 0f
        cpuHistory.clear()
        ramHistory.clear()
        latencyHistory.clear()
        throughputHistory.clear()
        if (isAttachedToWindow) {
            invalidate()
        }
    }
}
