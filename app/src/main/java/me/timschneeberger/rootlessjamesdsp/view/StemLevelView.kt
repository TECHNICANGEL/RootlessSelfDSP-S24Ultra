package me.timschneeberger.rootlessjamesdsp.view

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/**
 * S24 ULTRA - Stem Level Visualizer
 *
 * Shows real-time audio levels for each separated stem:
 * - Vocals (pink)
 * - Drums (gold)
 * - Bass (purple)
 * - Other (cyan)
 *
 * Horizontal bars with gradient fill and peak hold
 */
class StemLevelView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        private const val BAR_HEIGHT_RATIO = 0.18f
        private const val BAR_SPACING_RATIO = 0.06f
        private const val CORNER_RADIUS = 8f
        private const val PEAK_HOLD_TIME = 1500L  // ms
        private const val PEAK_DECAY = 0.02f
        private const val SMOOTHING = 0.3f
    }

    // Stem colors
    private val vocalsColor = Color.parseColor("#FF4081")  // Pink
    private val drumsColor = Color.parseColor("#FFD700")   // Gold
    private val bassColor = Color.parseColor("#9C27B0")    // Purple
    private val otherColor = Color.parseColor("#00E5FF")   // Cyan

    // Current levels (0-1)
    private var vocalsLevel = 0f
    private var drumsLevel = 0f
    private var bassLevel = 0f
    private var otherLevel = 0f

    // Smoothed levels for display
    private var vocalsSmooth = 0f
    private var drumsSmooth = 0f
    private var bassSmooth = 0f
    private var otherSmooth = 0f

    // Peak levels
    private var vocalsPeak = 0f
    private var drumsPeak = 0f
    private var bassPeak = 0f
    private var otherPeak = 0f

    // Peak timestamps
    private var vocalsPeakTime = 0L
    private var drumsPeakTime = 0L
    private var bassPeakTime = 0L
    private var otherPeakTime = 0L

    // Animation state
    private var isAnimating = false

    // Paints
    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#1A1A1A")
    }

    private val peakPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#808080")
        textSize = 28f
        textAlign = Paint.Align.LEFT
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    // Gradients (created in onSizeChanged)
    private var vocalsGradient: LinearGradient? = null
    private var drumsGradient: LinearGradient? = null
    private var bassGradient: LinearGradient? = null
    private var otherGradient: LinearGradient? = null

    private val rect = RectF()

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        createGradients(w.toFloat())
    }

    private fun createGradients(width: Float) {
        // Edge case: zero or negative width
        val barWidth = (width * 0.75f).coerceAtLeast(1f)

        try {
            vocalsGradient = LinearGradient(
                0f, 0f, barWidth, 0f,
                intArrayOf(vocalsColor.withAlpha(100), vocalsColor, vocalsColor.brighten()),
                floatArrayOf(0f, 0.7f, 1f),
                Shader.TileMode.CLAMP
            )

            drumsGradient = LinearGradient(
                0f, 0f, barWidth, 0f,
                intArrayOf(drumsColor.withAlpha(100), drumsColor, drumsColor.brighten()),
                floatArrayOf(0f, 0.7f, 1f),
                Shader.TileMode.CLAMP
            )

            bassGradient = LinearGradient(
                0f, 0f, barWidth, 0f,
                intArrayOf(bassColor.withAlpha(100), bassColor, bassColor.brighten()),
                floatArrayOf(0f, 0.7f, 1f),
                Shader.TileMode.CLAMP
            )

            otherGradient = LinearGradient(
                0f, 0f, barWidth, 0f,
                intArrayOf(otherColor.withAlpha(100), otherColor, otherColor.brighten()),
                floatArrayOf(0f, 0.7f, 1f),
                Shader.TileMode.CLAMP
            )
        } catch (e: Exception) {
            // Fallback: null gradients will use solid colors
            vocalsGradient = null
            drumsGradient = null
            bassGradient = null
            otherGradient = null
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Skip drawing if view is detached or surface invalid - prevents HWUI surface errors
        // Multiple checks needed because surface can be invalid even when view is attached
        if (!isAttachedToWindow || windowToken == null || display == null) {
            isAnimating = false
            return
        }

        val w = width.toFloat()
        val h = height.toFloat()

        // Skip drawing if view is too small or has invalid dimensions
        if (w < 50 || h < 50) {
            isAnimating = false
            return
        }

        val labelWidth = w * 0.12f
        val barStartX = labelWidth + 8f
        val barWidth = w - barStartX - 16f
        val barHeight = h * BAR_HEIGHT_RATIO
        val spacing = h * BAR_SPACING_RATIO

        val currentTime = System.currentTimeMillis()

        // Smooth the levels
        vocalsSmooth = lerp(vocalsSmooth, vocalsLevel, SMOOTHING)
        drumsSmooth = lerp(drumsSmooth, drumsLevel, SMOOTHING)
        bassSmooth = lerp(bassSmooth, bassLevel, SMOOTHING)
        otherSmooth = lerp(otherSmooth, otherLevel, SMOOTHING)

        // Update peaks
        updatePeak(vocalsSmooth, vocalsPeak, vocalsPeakTime, currentTime) { p, t ->
            vocalsPeak = p; vocalsPeakTime = t
        }
        updatePeak(drumsSmooth, drumsPeak, drumsPeakTime, currentTime) { p, t ->
            drumsPeak = p; drumsPeakTime = t
        }
        updatePeak(bassSmooth, bassPeak, bassPeakTime, currentTime) { p, t ->
            bassPeak = p; bassPeakTime = t
        }
        updatePeak(otherSmooth, otherPeak, otherPeakTime, currentTime) { p, t ->
            otherPeak = p; otherPeakTime = t
        }

        // Draw each stem bar
        var y = spacing

        // Vocals
        drawStemBar(canvas, "VOC", barStartX, y, barWidth, barHeight,
            vocalsSmooth, vocalsPeak, vocalsGradient, vocalsColor)
        y += barHeight + spacing

        // Drums
        drawStemBar(canvas, "DRM", barStartX, y, barWidth, barHeight,
            drumsSmooth, drumsPeak, drumsGradient, drumsColor)
        y += barHeight + spacing

        // Bass
        drawStemBar(canvas, "BAS", barStartX, y, barWidth, barHeight,
            bassSmooth, bassPeak, bassGradient, bassColor)
        y += barHeight + spacing

        // Other
        drawStemBar(canvas, "OTH", barStartX, y, barWidth, barHeight,
            otherSmooth, otherPeak, otherGradient, otherColor)

        // Continue animation
        if (isAnimating && isAttachedToWindow && windowToken != null) {
            postInvalidateOnAnimation()
        }
    }

    private fun drawStemBar(
        canvas: Canvas,
        label: String,
        x: Float,
        y: Float,
        width: Float,
        height: Float,
        level: Float,
        peak: Float,
        gradient: LinearGradient?,
        color: Int
    ) {
        // Label
        labelPaint.color = color.withAlpha(180)
        labelPaint.textSize = height * 0.6f
        canvas.drawText(label, 8f, y + height * 0.7f, labelPaint)

        // Background
        rect.set(x, y, x + width, y + height)
        canvas.drawRoundRect(rect, CORNER_RADIUS, CORNER_RADIUS, bgPaint)

        // Level bar
        val safeLevel = level.sanitize(0f).coerceIn(0f, 1f)
        val levelWidth = width * safeLevel
        if (levelWidth > 0.5f) {  // Minimum visible width
            rect.set(x, y, x + levelWidth, y + height)
            if (gradient != null) {
                barPaint.shader = gradient
            } else {
                // Fallback: solid color if gradient failed
                barPaint.shader = null
                barPaint.color = color
            }
            canvas.drawRoundRect(rect, CORNER_RADIUS, CORNER_RADIUS, barPaint)
            barPaint.shader = null
        }

        // Peak indicator
        val peakX = x + width * peak.coerceIn(0f, 1f)
        if (peak > 0.01f) {
            peakPaint.color = color
            canvas.drawRoundRect(
                peakX - 3f, y + 2f, peakX + 3f, y + height - 2f,
                2f, 2f, peakPaint
            )
        }
    }

    private inline fun updatePeak(
        level: Float,
        currentPeak: Float,
        peakTime: Long,
        currentTime: Long,
        update: (Float, Long) -> Unit
    ) {
        when {
            level > currentPeak -> update(level, currentTime)
            currentTime - peakTime > PEAK_HOLD_TIME -> {
                val newPeak = (currentPeak - PEAK_DECAY).coerceAtLeast(0f)
                update(newPeak, peakTime)
            }
        }
    }

    /**
     * Update stem levels (0-1 range)
     * Thread-safe and handles NaN/Infinity inputs
     */
    fun updateLevels(vocals: Float, drums: Float, bass: Float, other: Float) {
        // Sanitize NaN/Infinity inputs
        vocalsLevel = vocals.sanitize(0f).coerceIn(0f, 1f)
        drumsLevel = drums.sanitize(0f).coerceIn(0f, 1f)
        bassLevel = bass.sanitize(0f).coerceIn(0f, 1f)
        otherLevel = other.sanitize(0f).coerceIn(0f, 1f)

        // Only start animation if attached and not already animating
        if (!isAnimating && isAttachedToWindow) {
            isAnimating = true
            post { if (isAttachedToWindow) invalidate() }
        }
    }

    /**
     * Sanitize float - replace NaN/Infinity with default
     */
    private fun Float.sanitize(default: Float): Float {
        return if (this.isNaN() || this.isInfinite()) default else this
    }

    fun reset() {
        isAnimating = false
        vocalsLevel = 0f
        drumsLevel = 0f
        bassLevel = 0f
        otherLevel = 0f
        vocalsSmooth = 0f
        drumsSmooth = 0f
        bassSmooth = 0f
        otherSmooth = 0f
        vocalsPeak = 0f
        drumsPeak = 0f
        bassPeak = 0f
        otherPeak = 0f
        if (isAttachedToWindow) invalidate()
    }

    override fun onDetachedFromWindow() {
        isAnimating = false
        super.onDetachedFromWindow()
    }

    private fun lerp(current: Float, target: Float, factor: Float): Float {
        val result = current + (target - current) * factor
        return if (result.isNaN() || result.isInfinite()) current else result
    }

    private fun Int.withAlpha(alpha: Int): Int {
        return Color.argb(alpha, Color.red(this), Color.green(this), Color.blue(this))
    }

    private fun Int.brighten(): Int {
        val factor = 1.3f
        val r = (Color.red(this) * factor).toInt().coerceAtMost(255)
        val g = (Color.green(this) * factor).toInt().coerceAtMost(255)
        val b = (Color.blue(this) * factor).toInt().coerceAtMost(255)
        return Color.rgb(r, g, b)
    }
}
