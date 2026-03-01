package me.timschneeberger.rootlessjamesdsp.view

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.Choreographer
import android.view.View

/**
 * S24 ULTRA - Professional VU Meter
 *
 * Dual channel (L/R) VU meters with:
 * - Peak Program Meter (PPM) - Fast attack, slow release
 * - RMS meter - Average level
 * - Peak hold indicators
 * - Clip detection
 * - dB scale (-60 to +6 dB)
 *
 * Uses Choreographer for vsync-aligned 60/120fps animation
 */
class VUMeterView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Current display levels (interpolated)
    private var leftRms = -60f
    private var rightRms = -60f
    private var leftPeak = -60f
    private var rightPeak = -60f
    private var leftPeakHold = -60f
    private var rightPeakHold = -60f

    // Target levels (from audio data)
    @Volatile private var targetLeftRms = -60f
    @Volatile private var targetRightRms = -60f
    @Volatile private var targetLeftPeak = -60f
    @Volatile private var targetRightPeak = -60f

    // Clip detection
    private var leftClipped = false
    private var rightClipped = false
    private var clipResetTime = 0L

    // Choreographer for vsync-aligned rendering
    private var choreographer: Choreographer? = null
    private var isAnimating = false
    private var lastDataUpdate = 0L
    private val animationTimeout = 2000L

    // Ballistics - smooth interpolation per frame
    private val attackFactor = 0.3f  // Fast attack
    private val releaseFactor = 0.08f // Smooth release
    private val peakHoldTime = 800L

    // Colors - Professional studio style with glow
    private val colorGreen = Color.parseColor("#00FF88")   // -20 to -6 dB (ultra green)
    private val colorYellow = Color.parseColor("#FFD700")  // -6 to 0 dB (gold)
    private val colorOrange = Color.parseColor("#FF8800")  // 0 to +3 dB
    private val colorRed = Color.parseColor("#FF1744")     // +3 to +6 dB (danger)
    private val colorClip = Color.parseColor("#FF0080")    // Clip indicator (magenta)
    private val colorIdle = Color.parseColor("#333333")    // Idle state

    private val bgPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#0A0A0A")
    }

    private val meterPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CCCCCC")
        textSize = 28f
        typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
    }

    private val peakPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.parseColor("#FFD700") // Gold
    }

    // No signal indicator
    private var showNoSignal = true

    // Choreographer frame callback - interpolates values each frame
    private val frameCallback = object : Choreographer.FrameCallback {
        override fun doFrame(frameTimeNanos: Long) {
            if (!isAnimating || !isAttachedToWindow || windowToken == null) {
                return
            }

            // Check timeout
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastDataUpdate > animationTimeout) {
                isAnimating = false
                return
            }

            // Interpolate current values towards targets
            leftRms = lerp(leftRms, targetLeftRms)
            rightRms = lerp(rightRms, targetRightRms)
            leftPeak = lerp(leftPeak, targetLeftPeak)
            rightPeak = lerp(rightPeak, targetRightPeak)

            // Update peak hold
            if (targetLeftPeak > leftPeakHold) {
                leftPeakHold = targetLeftPeak
                leftPeakHoldTime = currentTime
            } else if (currentTime - leftPeakHoldTime > peakHoldTime) {
                leftPeakHold = (leftPeakHold - 0.5f).coerceAtLeast(-60f)
            }

            if (targetRightPeak > rightPeakHold) {
                rightPeakHold = targetRightPeak
                rightPeakHoldTime = currentTime
            } else if (currentTime - rightPeakHoldTime > peakHoldTime) {
                rightPeakHold = (rightPeakHold - 0.5f).coerceAtLeast(-60f)
            }

            // Clip detection
            if (targetLeftPeak >= 6f) {
                leftClipped = true
                clipResetTime = currentTime
            }
            if (targetRightPeak >= 6f) {
                rightClipped = true
                clipResetTime = currentTime
            }
            if (currentTime - clipResetTime > 3000) {
                leftClipped = false
                rightClipped = false
            }

            // Trigger redraw
            invalidate()

            // Schedule next frame
            choreographer?.postFrameCallback(this)
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        choreographer = Choreographer.getInstance()

        // Restart animation if we have pending data (view was reattached)
        if (lastDataUpdate > 0 && System.currentTimeMillis() - lastDataUpdate < animationTimeout) {
            startAnimation()
        }
    }

    override fun onVisibilityChanged(changedView: View, visibility: Int) {
        super.onVisibilityChanged(changedView, visibility)
        when (visibility) {
            VISIBLE -> {
                // Restart animation if we have recent data
                if (lastDataUpdate > 0 && System.currentTimeMillis() - lastDataUpdate < animationTimeout) {
                    startAnimation()
                }
            }
            INVISIBLE, GONE -> {
                // Stop animation to save resources when not visible
                stopAnimation()
            }
        }
    }

    override fun onWindowVisibilityChanged(visibility: Int) {
        super.onWindowVisibilityChanged(visibility)
        when (visibility) {
            VISIBLE -> {
                if (lastDataUpdate > 0 && System.currentTimeMillis() - lastDataUpdate < animationTimeout) {
                    startAnimation()
                }
            }
            INVISIBLE, GONE -> {
                stopAnimation()
            }
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (!isAttachedToWindow || windowToken == null || display == null) {
            return
        }

        val width = width.toFloat()
        val height = height.toFloat()

        if (width < 100f || height < 100f) {
            return
        }

        // Background
        canvas.drawRoundRect(0f, 0f, width, height, 16f, 16f, bgPaint)

        // Calculate meter dimensions
        val meterHeight = ((height - 100f) / 2f - 20f).coerceAtLeast(20f)
        val meterWidth = (width - 120f).coerceAtLeast(50f)
        val meterX = 80f.coerceAtMost(width * 0.2f)

        showNoSignal = leftRms <= -59f && rightRms <= -59f

        // Draw channels
        drawChannel(canvas, meterX, 40f, meterWidth, meterHeight,
            leftRms, leftPeak, leftPeakHold, leftClipped, "L")
        drawChannel(canvas, meterX, height / 2f + 20f, meterWidth, meterHeight,
            rightRms, rightPeak, rightPeakHold, rightClipped, "R")

        drawScale(canvas, meterX, meterWidth)

        if (showNoSignal && !isAnimating) {
            drawNoSignalIndicator(canvas, width, height)
        }
    }

    override fun onDetachedFromWindow() {
        stopAnimation()
        choreographer = null
        super.onDetachedFromWindow()
    }

    private fun startAnimation() {
        if (isAnimating) return
        isAnimating = true
        choreographer?.postFrameCallback(frameCallback)
    }

    private fun stopAnimation() {
        isAnimating = false
        choreographer?.removeFrameCallback(frameCallback)
    }

    private fun drawChannel(
        canvas: Canvas, x: Float, y: Float, width: Float, height: Float,
        rms: Float, peak: Float, peakHold: Float, clipped: Boolean, label: String
    ) {
        // Channel label
        textPaint.textAlign = Paint.Align.RIGHT
        textPaint.textSize = 32f
        textPaint.color = if (clipped) colorClip else Color.WHITE
        canvas.drawText(label, x - 10f, y + height / 2f + 12f, textPaint)

        // Background bar
        meterPaint.color = Color.parseColor("#1A1A1A")
        canvas.drawRoundRect(x, y, x + width, y + height, 8f, 8f, meterPaint)

        // RMS meter (main indicator)
        val rmsWidth = width * dbToNormalized(rms)
        drawGradientBar(canvas, x, y, rmsWidth, height, rms, 0.7f)

        // Peak meter (brighter, thinner)
        val peakWidth = width * dbToNormalized(peak)
        drawGradientBar(canvas, x, y + height * 0.3f, peakWidth, height * 0.4f, peak, 1f)

        // Peak hold line
        if (peakHold > -60f) {
            val peakHoldX = x + width * dbToNormalized(peakHold)
            peakPaint.color = colorClip
            canvas.drawLine(peakHoldX, y, peakHoldX, y + height, peakPaint)
        }

        // Clip indicator with dramatic effect
        if (clipped) {
            // Flashing clip indicator
            val flashAlpha = ((System.currentTimeMillis() % 500) / 500f * 255).toInt()

            // Red clip box
            meterPaint.color = colorClip
            meterPaint.alpha = 200
            canvas.drawRoundRect(x + width - 50f, y, x + width, y + height, 8f, 8f, meterPaint)

            // "CLIP" text
            textPaint.color = Color.WHITE
            textPaint.textSize = 14f
            textPaint.textAlign = Paint.Align.CENTER
            canvas.drawText("!", x + width - 25f, y + height / 2f + 5f, textPaint)

            // Pulsing border around entire meter
            meterPaint.style = Paint.Style.STROKE
            meterPaint.strokeWidth = 3f
            meterPaint.color = colorClip
            meterPaint.alpha = flashAlpha
            canvas.drawRoundRect(x - 2f, y - 2f, x + width + 2f, y + height + 2f, 10f, 10f, meterPaint)
            meterPaint.style = Paint.Style.FILL
            meterPaint.alpha = 255
        }
    }

    private fun drawGradientBar(canvas: Canvas, x: Float, y: Float, width: Float, height: Float, db: Float, alpha: Float) {
        if (width <= 0) return

        // Color based on dB level
        val color = when {
            db >= 3f -> colorRed
            db >= 0f -> colorOrange
            db >= -6f -> colorYellow
            else -> colorGreen
        }

        // Draw glow effect for high levels
        // Note: BlurMaskFilter only works with software rendering, but we can't call
        // setLayerType during onDraw as it causes HWUI surface errors. Instead, draw
        // a simple semi-transparent background for the glow effect.
        if (db > -20f && alpha > 0.8f) {
            // Simple glow simulation without blur filter (hardware-accelerated safe)
            meterPaint.color = color
            meterPaint.alpha = (60 * alpha).toInt()
            canvas.drawRoundRect(x - 4f, y - 4f, x + width + 4f, y + height + 4f, 12f, 12f, meterPaint)
            meterPaint.alpha = (40 * alpha).toInt()
            canvas.drawRoundRect(x - 8f, y - 8f, x + width + 8f, y + height + 8f, 16f, 16f, meterPaint)
        }

        // Draw main bar
        meterPaint.color = color
        meterPaint.alpha = (255 * alpha).toInt()
        canvas.drawRoundRect(x, y, x + width, y + height, 8f, 8f, meterPaint)
        meterPaint.alpha = 255
    }

    /**
     * Draw "NO SIGNAL" indicator when idle
     */
    private fun drawNoSignalIndicator(canvas: Canvas, width: Float, height: Float) {
        textPaint.textAlign = Paint.Align.CENTER
        textPaint.textSize = 24f
        textPaint.color = colorIdle
        canvas.drawText("NO SIGNAL", width / 2f, height / 2f, textPaint)
    }

    private fun drawScale(canvas: Canvas, meterX: Float, meterWidth: Float) {
        textPaint.textAlign = Paint.Align.CENTER
        textPaint.textSize = 16f
        textPaint.color = Color.parseColor("#666666")

        val markers = listOf(-60, -40, -20, -10, -6, -3, 0, 3, 6)
        markers.forEach { db ->
            val x = meterX + meterWidth * dbToNormalized(db.toFloat())
            // Draw small tick
            canvas.drawLine(x, 10f, x, 20f, textPaint)
        }
    }

    private fun dbToNormalized(db: Float): Float {
        return ((db + 60f) / 66f).coerceIn(0f, 1f)
    }

    // Peak hold timestamps for decay
    private var leftPeakHoldTime = 0L
    private var rightPeakHoldTime = 0L

    /**
     * Update audio levels - sets targets for smooth interpolation
     * Choreographer handles the animation at display refresh rate
     */
    fun updateLevels(leftRms: Float, rightRms: Float, leftPeak: Float, rightPeak: Float) {
        lastDataUpdate = System.currentTimeMillis()

        // Set targets (Choreographer will interpolate towards these)
        targetLeftRms = leftRms.sanitize(-60f).coerceIn(-60f, 10f)
        targetRightRms = rightRms.sanitize(-60f).coerceIn(-60f, 10f)
        targetLeftPeak = leftPeak.sanitize(-60f).coerceIn(-60f, 10f)
        targetRightPeak = rightPeak.sanitize(-60f).coerceIn(-60f, 10f)

        // Start animation if not running
        if (!isAnimating && isAttachedToWindow) {
            startAnimation()
        }
    }

    /**
     * Sanitize float value - replace NaN/Infinity with default
     */
    private fun Float.sanitize(default: Float): Float {
        return if (this.isNaN() || this.isInfinite()) default else this
    }

    private fun lerp(current: Float, target: Float): Float {
        val delta = target - current
        return if (delta > 0) {
            // Attack - fast response (85% towards target per frame)
            current + delta * attackFactor
        } else {
            // Release - smooth decay (15% towards target per frame)
            current + delta * releaseFactor
        }
    }

    /**
     * Reset meters
     */
    fun reset() {
        stopAnimation()
        leftRms = -60f
        rightRms = -60f
        leftPeak = -60f
        rightPeak = -60f
        leftPeakHold = -60f
        rightPeakHold = -60f
        targetLeftRms = -60f
        targetRightRms = -60f
        targetLeftPeak = -60f
        targetRightPeak = -60f
        leftPeakHoldTime = 0L
        rightPeakHoldTime = 0L
        leftClipped = false
        rightClipped = false
        clipResetTime = 0L
        if (isAttachedToWindow) {
            invalidate()
        }
    }
}
