package me.timschneeberger.rootlessjamesdsp.view

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.DecelerateInterpolator

/**
 * User-Friendly Performance Monitor
 *
 * Simplified view showing:
 * - Speedometer gauge with color-coded performance
 * - Simple "Fast/Normal/Slow" label
 * - Compact stem timing bars
 */
class NpuStatsView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Stats data
    private var npuUsagePercent = 0f
    private var realtimeFactor = 0f
    private var stftMs = 0
    private var npuMs = 0
    private var istftMs = 0
    private var totalMs = 0
    private var queueSize = 0
    private var state = "IDLE"

    // Individual model stats
    private var vocalsMs = 0
    private var drumsMs = 0
    private var bassMs = 0
    private var otherMs = 0
    private var usingNpu = false
    private var backendName = "CPU"

    // Animated values for smooth transitions
    private var animatedRtFactor = 0f
    private var gaugeAnimator: ValueAnimator? = null

    // Colors
    private val colorSuccess = Color.parseColor("#00E676")    // Green
    private val colorWarning = Color.parseColor("#FFD600")    // Yellow
    private val colorDanger = Color.parseColor("#FF5252")     // Red
    private val colorPrimary = Color.parseColor("#00E5FF")    // Cyan
    private val colorBg = Color.parseColor("#0D0D0D")
    private val colorBgLight = Color.parseColor("#1A1A1A")
    private val colorText = Color.parseColor("#E0E0E0")
    private val colorTextDim = Color.parseColor("#757575")

    // Stem colors
    private val colorVocals = Color.parseColor("#E040FB")
    private val colorDrums = Color.parseColor("#FF9800")
    private val colorBass = Color.parseColor("#00E5FF")
    private val colorOther = Color.parseColor("#4CAF50")

    // Paints
    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = colorBg
    }

    private val gaugeBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 12f
        strokeCap = Paint.Cap.ROUND
        color = colorBgLight
    }

    private val gaugePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 12f
        strokeCap = Paint.Cap.ROUND
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorText
        textSize = 14f
        textAlign = Paint.Align.CENTER
    }

    private val valuePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 28f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private val smallPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorTextDim
        textSize = 11f
        textAlign = Paint.Align.LEFT
    }

    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val rect = RectF()
    private val gaugeRect = RectF()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        val padding = 16f

        // Background
        rect.set(0f, 0f, w, h)
        bgPaint.color = colorBg
        canvas.drawRoundRect(rect, 16f, 16f, bgPaint)

        // === TOP ROW: Performance Gauge + Speed Label ===
        val gaugeSize = 80f
        val gaugeCenterX = padding + gaugeSize / 2 + 8f
        val gaugeCenterY = padding + gaugeSize / 2

        // Draw gauge background arc (180 degrees, bottom half)
        gaugeRect.set(
            gaugeCenterX - gaugeSize / 2 + 6f,
            gaugeCenterY - gaugeSize / 2 + 6f,
            gaugeCenterX + gaugeSize / 2 - 6f,
            gaugeCenterY + gaugeSize / 2 - 6f
        )
        canvas.drawArc(gaugeRect, 180f, 180f, false, gaugeBgPaint)

        // Draw gauge progress arc
        val gaugeColor = when {
            animatedRtFactor >= 2f -> colorSuccess
            animatedRtFactor >= 1f -> colorWarning
            else -> colorDanger
        }
        gaugePaint.color = gaugeColor

        // Map RT factor to angle (0 = empty, 3+ = full)
        val sweepAngle = (animatedRtFactor / 3f).coerceIn(0f, 1f) * 180f
        canvas.drawArc(gaugeRect, 180f, sweepAngle, false, gaugePaint)

        // Speed value in center
        valuePaint.color = gaugeColor
        valuePaint.textSize = 20f
        val speedLabel = when {
            animatedRtFactor >= 3f -> "TURBO"
            animatedRtFactor >= 2f -> "FAST"
            animatedRtFactor >= 1f -> "OK"
            animatedRtFactor >= 0.5f -> "SLOW"
            animatedRtFactor > 0f -> "LAG"
            else -> "—"
        }
        canvas.drawText(speedLabel, gaugeCenterX, gaugeCenterY + 6f, valuePaint)

        // RT factor below gauge
        smallPaint.textAlign = Paint.Align.CENTER
        smallPaint.color = colorTextDim
        canvas.drawText(
            if (animatedRtFactor > 0) String.format("%.1fx realtime", animatedRtFactor) else "Waiting...",
            gaugeCenterX,
            gaugeCenterY + gaugeSize / 2 + 4f,
            smallPaint
        )

        // === RIGHT SIDE: Stats Summary ===
        val statsX = gaugeCenterX + gaugeSize / 2 + 24f
        var statsY = padding + 16f

        // Backend label - color based on type
        val backendColor = when {
            backendName.contains("NPU", ignoreCase = true) ||
            backendName.contains("Hexagon", ignoreCase = true) ||
            backendName.contains("HTP", ignoreCase = true) -> colorSuccess
            backendName.contains("NNAPI", ignoreCase = true) -> colorPrimary
            backendName.contains("GPU", ignoreCase = true) -> colorPrimary
            else -> colorWarning  // CPU
        }
        labelPaint.textAlign = Paint.Align.LEFT
        labelPaint.color = backendColor
        labelPaint.textSize = 12f
        canvas.drawText(backendName, statsX, statsY, labelPaint)

        statsY += 24f

        // Total processing time
        labelPaint.color = colorText
        labelPaint.textSize = 14f
        canvas.drawText("Processing: ${totalMs}ms", statsX, statsY, labelPaint)

        statsY += 20f

        // Queue status
        val queueColor = when {
            queueSize >= 5 -> colorSuccess
            queueSize >= 2 -> colorWarning
            else -> colorDanger
        }
        labelPaint.color = queueColor
        canvas.drawText("Buffer: $queueSize chunks", statsX, statsY, labelPaint)

        // === BOTTOM: Stem Processing Bars ===
        val barSectionY = padding + gaugeSize + 16f
        val barHeight = 8f
        val barSpacing = 20f
        val labelWidth = 50f
        val barWidth = w - padding * 2 - labelWidth - 50f

        // Calculate max time for scaling
        val maxTime = maxOf(vocalsMs, drumsMs, bassMs, otherMs, 1)
        val chunkMs = 2000 // 2 second chunks

        val stems = listOf(
            Triple("Vocals", vocalsMs, colorVocals),
            Triple("Drums", drumsMs, colorDrums),
            Triple("Bass", bassMs, colorBass),
            Triple("Other", otherMs, colorOther)
        )

        var barY = barSectionY

        for ((name, ms, color) in stems) {
            // Label
            smallPaint.textAlign = Paint.Align.LEFT
            smallPaint.color = color
            canvas.drawText(name, padding, barY + barHeight, smallPaint)

            // Bar background
            val barX = padding + labelWidth
            barPaint.color = colorBgLight
            rect.set(barX, barY, barX + barWidth, barY + barHeight)
            canvas.drawRoundRect(rect, 4f, 4f, barPaint)

            // Bar fill (relative to chunk time for realtime comparison)
            val fillPct = (ms.toFloat() / (chunkMs / 4)).coerceIn(0f, 1f) // Each stem should be <500ms
            barPaint.color = color
            barPaint.alpha = 200
            rect.set(barX, barY, barX + barWidth * fillPct, barY + barHeight)
            canvas.drawRoundRect(rect, 4f, 4f, barPaint)
            barPaint.alpha = 255

            // Time value
            smallPaint.textAlign = Paint.Align.RIGHT
            smallPaint.color = colorTextDim
            canvas.drawText("${ms}ms", w - padding, barY + barHeight, smallPaint)

            barY += barSpacing
        }
    }

    fun updateStats(
        npuUsage: Float,
        rtFactor: Float,
        stftTime: Int,
        npuTime: Int,
        istftTime: Int,
        queue: Int,
        currentState: String,
        bufferPct: Int = 0
    ) {
        this.npuUsagePercent = npuUsage.coerceIn(0f, 200f)
        this.stftMs = stftTime
        this.npuMs = npuTime
        this.istftMs = istftTime
        this.totalMs = stftTime + npuTime + istftTime
        this.queueSize = queue
        this.state = currentState

        // Animate RT factor change
        animateToRtFactor(rtFactor.coerceIn(0f, 5f))
    }

    private fun animateToRtFactor(target: Float) {
        gaugeAnimator?.cancel()
        gaugeAnimator = ValueAnimator.ofFloat(animatedRtFactor, target).apply {
            duration = 300
            interpolator = DecelerateInterpolator()
            addUpdateListener {
                animatedRtFactor = it.animatedValue as Float
                invalidate()
            }
            start()
        }
    }

    fun updateModelStats(
        vocalsTime: Int,
        drumsTime: Int,
        bassTime: Int,
        otherTime: Int,
        totalTime: Int,
        isUsingNpu: Boolean,
        backend: String = "CPU"
    ) {
        this.vocalsMs = vocalsTime
        this.drumsMs = drumsTime
        this.bassMs = bassTime
        this.otherMs = otherTime
        this.usingNpu = isUsingNpu
        this.backendName = backend
        invalidate()
    }

    fun reset() {
        gaugeAnimator?.cancel()
        npuUsagePercent = 0f
        realtimeFactor = 0f
        animatedRtFactor = 0f
        stftMs = 0
        npuMs = 0
        istftMs = 0
        totalMs = 0
        queueSize = 0
        state = "IDLE"
        vocalsMs = 0
        drumsMs = 0
        bassMs = 0
        otherMs = 0
        usingNpu = false
        invalidate()
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        gaugeAnimator?.cancel()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val desiredWidth = 400
        val desiredHeight = 180  // Much more compact!

        val width = when (MeasureSpec.getMode(widthMeasureSpec)) {
            MeasureSpec.EXACTLY -> MeasureSpec.getSize(widthMeasureSpec)
            MeasureSpec.AT_MOST -> minOf(desiredWidth, MeasureSpec.getSize(widthMeasureSpec))
            else -> desiredWidth
        }

        val height = when (MeasureSpec.getMode(heightMeasureSpec)) {
            MeasureSpec.EXACTLY -> MeasureSpec.getSize(heightMeasureSpec)
            MeasureSpec.AT_MOST -> minOf(desiredHeight, MeasureSpec.getSize(heightMeasureSpec))
            else -> desiredHeight
        }

        setMeasuredDimension(width, height)
    }
}
