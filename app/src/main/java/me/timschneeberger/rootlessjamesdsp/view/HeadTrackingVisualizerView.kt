package me.timschneeberger.rootlessjamesdsp.view

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.sin

/**
 * S24 ULTRA + XM5 - Head Tracking Visualizer
 *
 * Real-time 3D visualization of head position from Sony WH-1000XM5
 * Shows yaw/pitch/roll for spatial audio debugging
 *
 * Renders at 60fps with smooth interpolation
 */
class HeadTrackingVisualizerView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // PERFORMANCE: Pre-cached colors to avoid Color.parseColor() in onDraw
    companion object {
        private val COLOR_WARNING = Color.parseColor("#FF8800")
        private val COLOR_NORMAL = Color.parseColor("#00FF88")
        private val COLOR_TITLE = Color.parseColor("#9C27B0")
        private val COLOR_GRID = Color.parseColor("#333333")
    }

    // Head pose (degrees)
    private var yaw = 0f      // Left/Right rotation
    private var pitch = 0f    // Up/Down tilt
    private var roll = 0f     // Left/Right tilt

    // Smoothing - RAW POWER: Faster response
    private var targetYaw = 0f
    private var targetPitch = 0f
    private var targetRoll = 0f
    private val smoothing = 0.3f // RAW POWER: Less smoothing, more responsive

    // Animation state - RAW POWER: Keep animating
    private var isAnimating = false
    private var lastDataUpdate = 0L
    private val animationTimeout = 3000L // RAW POWER: Keep animating 3s

    // Paint objects
    private val headPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = COLOR_TITLE
    }

    private val outlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.WHITE
    }

    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1f
        color = COLOR_GRID
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 32f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#808080")
        textSize = 20f
        textAlign = Paint.Align.CENTER
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

        val centerX = width / 2f
        val centerY = height / 2f

        // Background
        canvas.drawColor(Color.BLACK)

        // Smooth interpolation
        yaw = lerp(yaw, targetYaw, smoothing)
        pitch = lerp(pitch, targetPitch, smoothing)
        roll = lerp(roll, targetRoll, smoothing)

        // Draw grid
        drawGrid(canvas, centerX, centerY, width, height)

        // Draw head representation (simple circle with direction indicator)
        drawHead(canvas, centerX, centerY, yaw, pitch, roll)

        // Draw angles
        drawAngles(canvas, width, height)

        // Draw status
        drawStatus(canvas, width, height)

        // Only continue animation if we're receiving data and view is attached with valid surface
        // This prevents idle CPU/battery usage when head tracking is inactive
        // and HWUI errors when rendering to destroyed surfaces
        val currentTime = System.currentTimeMillis()
        if (isAnimating && currentTime - lastDataUpdate < animationTimeout && isAttachedToWindow && windowToken != null) {
            postInvalidateOnAnimation()
        } else {
            isAnimating = false
        }
    }

    override fun onDetachedFromWindow() {
        isAnimating = false
        super.onDetachedFromWindow()
    }

    private fun drawGrid(canvas: Canvas, cx: Float, cy: Float, width: Float, height: Float) {
        // Horizon line
        gridPaint.color = Color.parseColor("#1A1A1A")
        canvas.drawLine(0f, cy, width, cy, gridPaint)
        canvas.drawLine(cx, 0f, cx, height, gridPaint)

        // Circles for reference
        for (radius in listOf(50f, 100f, 150f)) {
            canvas.drawCircle(cx, cy, radius, gridPaint)
        }
    }

    private fun drawHead(canvas: Canvas, cx: Float, cy: Float, yaw: Float, pitch: Float, roll: Float) {
        val headRadius = 80f

        canvas.save()
        canvas.translate(cx, cy)

        // Apply rotations for 2D representation
        canvas.rotate(roll)

        // Draw head circle
        headPaint.alpha = 200
        canvas.drawCircle(0f, 0f, headRadius, headPaint)
        headPaint.alpha = 255

        // Draw outline
        canvas.drawCircle(0f, 0f, headRadius, outlinePaint)

        // Draw direction indicator (nose)
        val noseLength = headRadius * 1.5f
        val noseX = sin(Math.toRadians(yaw.toDouble())).toFloat() * noseLength
        val noseY = -sin(Math.toRadians(pitch.toDouble())).toFloat() * noseLength

        outlinePaint.strokeWidth = 6f
        outlinePaint.color = Color.parseColor("#FFD700") // Gold
        canvas.drawLine(0f, 0f, noseX, noseY, outlinePaint)
        outlinePaint.color = Color.WHITE
        outlinePaint.strokeWidth = 4f

        // Draw ears (L/R indicators)
        val earRadius = 15f
        textPaint.textSize = 24f
        canvas.drawCircle(-headRadius - 20f, 0f, earRadius, headPaint)
        canvas.drawText("L", -headRadius - 20f, 8f, textPaint)

        canvas.drawCircle(headRadius + 20f, 0f, earRadius, headPaint)
        canvas.drawText("R", headRadius + 20f, 8f, textPaint)

        canvas.restore()
    }

    private fun drawAngles(canvas: Canvas, width: Float, height: Float) {
        // Skip if not enough space for angle display
        if (height < 200f) return

        val y = (height - 120f).coerceAtLeast(100f)

        labelPaint.textAlign = Paint.Align.CENTER

        // PERFORMANCE: Use pre-cached colors instead of Color.parseColor()
        // Yaw
        canvas.drawText("YAW", width * 0.2f, y, labelPaint)
        textPaint.color = if (kotlin.math.abs(yaw) > 45f) COLOR_WARNING else COLOR_NORMAL
        canvas.drawText("${yaw.toInt()}°", width * 0.2f, y + 35f, textPaint)

        // Pitch
        canvas.drawText("PITCH", width * 0.5f, y, labelPaint)
        textPaint.color = if (kotlin.math.abs(pitch) > 30f) COLOR_WARNING else COLOR_NORMAL
        canvas.drawText("${pitch.toInt()}°", width * 0.5f, y + 35f, textPaint)

        // Roll
        canvas.drawText("ROLL", width * 0.8f, y, labelPaint)
        textPaint.color = if (kotlin.math.abs(roll) > 15f) COLOR_WARNING else COLOR_NORMAL
        canvas.drawText("${roll.toInt()}°", width * 0.8f, y + 35f, textPaint)
    }

    private fun drawStatus(canvas: Canvas, width: Float, height: Float) {
        textPaint.textSize = 24f
        textPaint.color = COLOR_TITLE
        canvas.drawText("SONY WH-1000XM5 • HEAD TRACKING ACTIVE", width / 2f, 40f, textPaint)
    }

    /**
     * Update head pose from Spatializer or sensors
     */
    fun updateHeadPose(yaw: Float, pitch: Float, roll: Float) {
        // Update animation state
        lastDataUpdate = System.currentTimeMillis()
        val wasAnimating = isAnimating
        isAnimating = true

        // Sanitize and constrain inputs
        this.targetYaw = yaw.sanitize(0f).coerceIn(-180f, 180f)
        this.targetPitch = pitch.sanitize(0f).coerceIn(-90f, 90f)
        this.targetRoll = roll.sanitize(0f).coerceIn(-90f, 90f)

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

    private fun lerp(current: Float, target: Float, factor: Float): Float {
        val result = current + (target - current) * factor
        // Protect against NaN propagation
        return if (result.isNaN() || result.isInfinite()) current else result
    }

    /**
     * Reset to center
     */
    fun reset() {
        isAnimating = false
        // Reset both current and target values
        yaw = 0f
        pitch = 0f
        roll = 0f
        targetYaw = 0f
        targetPitch = 0f
        targetRoll = 0f
        if (isAttachedToWindow) {
            invalidate()
        }
    }
}
