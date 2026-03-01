package me.timschneeberger.rootlessjamesdsp.view

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import me.timschneeberger.rootlessjamesdsp.audio.SourceSeparator

/**
 * Train Pipeline Visualization
 *
 * Shows a train moving through stations:
 * [BUFFER] → [STFT] → [AI] → [ISTFT] → [OUTPUT]
 *
 * The train collects audio cargo and delivers it through each processing stage.
 * Single smooth animation - no flickering.
 */
class SeparationStateView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Current state
    private var currentState = SourceSeparator.SeparationState.IDLE
    private var previousState = SourceSeparator.SeparationState.IDLE
    private var progress = 0f
    private var hasCompletedFirstPass = false  // Track if we've done initial processing

    // Current train position (0-4 = at station, decimals = between stations)
    private var trainPosition = 0f
    private var targetTrainPosition = 0f

    // Animation
    private var wheelRotation = 0f
    private var smokeOffset = 0f
    private var cargoLevel = 0f // 0-1 how full the cargo car is
    private var deliveryPulse = 0f // For output station pulsing

    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 50
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            // Smooth train movement toward target (only move forward, not backward)
            val diff = targetTrainPosition - trainPosition
            if (diff > 0) {
                trainPosition += diff * 0.12f // Smooth lerp forward
            } else if (diff < -2f) {
                // Only reset if going way back (like IDLE reset)
                trainPosition += diff * 0.08f
            }
            // Otherwise stay put - don't jitter back

            // Wheel rotation (faster when moving)
            val speed = kotlin.math.abs(diff).coerceAtMost(1f)
            wheelRotation = (wheelRotation + 4f + speed * 15f) % 360f

            // Smoke animation
            smokeOffset = (smokeOffset + 1.5f) % 100f

            // Delivery pulse for output
            deliveryPulse = (deliveryPulse + 3f) % 360f

            invalidate()
        }
    }

    // Colors
    private val colorTrack = Color.parseColor("#333333")
    private val colorTrackTie = Color.parseColor("#1A1A1A")
    private val colorTrain = Color.parseColor("#00E5FF")      // Cyan engine
    private val colorCargo = Color.parseColor("#E040FB")      // Purple cargo
    private val colorWheel = Color.parseColor("#888888")
    private val colorSmoke = Color.parseColor("#666666")
    private val colorStation = Color.parseColor("#1A1A1A")
    private val colorStationActive = Color.parseColor("#00E5FF")
    private val colorBg = Color.parseColor("#0D0D0D")

    private val stationColors = listOf(
        Color.parseColor("#FFC107"),    // Buffer - Amber
        Color.parseColor("#2196F3"),    // STFT - Blue
        Color.parseColor("#E040FB"),    // AI - Purple
        Color.parseColor("#00BCD4"),    // ISTFT - Cyan
        Color.parseColor("#4CAF50")     // Output - Green
    )

    // Paints
    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorBg
        style = Paint.Style.FILL
    }

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorTrack
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val tiePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorTrackTie
        style = Paint.Style.FILL
    }

    private val trainPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorTrain
        style = Paint.Style.FILL
    }

    private val cargoPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorCargo
        style = Paint.Style.FILL
    }

    private val wheelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorWheel
        style = Paint.Style.FILL
    }

    private val smokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorSmoke
        style = Paint.Style.FILL
    }

    private val stationPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val stationBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 10f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private val statusPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 14f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private val rect = RectF()

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        animator.start()
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        animator.cancel()
    }

    fun updateState(state: SourceSeparator.SeparationState, progress: Float, message: String, npuUsage: Float = 0f) {
        this.previousState = this.currentState
        this.currentState = state
        this.progress = progress.coerceIn(0f, 1f)

        // Track when we first reach PLAYING
        if (state == SourceSeparator.SeparationState.PLAYING) {
            hasCompletedFirstPass = true
        }

        // Reset tracking if we go back to IDLE
        if (state == SourceSeparator.SeparationState.IDLE) {
            hasCompletedFirstPass = false
        }

        // Once we've started playing, keep train at output - don't show individual chunk processing
        if (hasCompletedFirstPass && state != SourceSeparator.SeparationState.IDLE &&
            state != SourceSeparator.SeparationState.EXHAUSTED) {
            // Stay at output, show continuous delivery
            targetTrainPosition = 4f
            cargoLevel = 0.6f
            return
        }

        // Initial processing - show train journey through stations
        when (state) {
            SourceSeparator.SeparationState.IDLE -> {
                targetTrainPosition = 0f
                cargoLevel = 0f
            }
            SourceSeparator.SeparationState.BUFFERING -> {
                targetTrainPosition = progress * 0.9f  // Filling up at buffer station
                cargoLevel = progress
            }
            SourceSeparator.SeparationState.PROCESSING_STFT -> {
                targetTrainPosition = 1f + progress * 0.9f  // At STFT station
                cargoLevel = 1f
            }
            SourceSeparator.SeparationState.PROCESSING_NPU -> {
                targetTrainPosition = 2f + progress * 0.9f  // At AI station
                cargoLevel = 1f
            }
            SourceSeparator.SeparationState.PROCESSING_ISTFT -> {
                targetTrainPosition = 3f + progress * 0.9f  // At ISTFT station
                cargoLevel = 1f
            }
            SourceSeparator.SeparationState.PLAYING -> {
                targetTrainPosition = 4f  // At output, delivering
                cargoLevel = 0.5f
            }
            SourceSeparator.SeparationState.EXHAUSTED -> {
                // Don't reset train, just show waiting state
                targetTrainPosition = 4f  // Stay at output
                cargoLevel = 0.1f
            }
        }
    }

    fun updatePipelineStats(bufferPct: Int, stftMs: Int, npuMs: Int, istftMs: Int, queueSize: Int) {
        // Stats can be shown if needed
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        val padding = 16f

        // Background
        rect.set(0f, 0f, w, h)
        canvas.drawRoundRect(rect, 16f, 16f, bgPaint)

        // Layout
        val trackY = h * 0.55f
        val stationY = h * 0.35f
        val stationRadius = 18f
        val stationSpacing = (w - padding * 2) / 4f

        // Draw railroad ties
        val tieWidth = 8f
        val tieHeight = 20f
        var tieX = padding
        while (tieX < w - padding) {
            rect.set(tieX - tieWidth/2, trackY - tieHeight/2, tieX + tieWidth/2, trackY + tieHeight/2)
            canvas.drawRect(rect, tiePaint)
            tieX += 20f
        }

        // Draw track rails
        trackPaint.strokeWidth = 3f
        canvas.drawLine(padding, trackY - 6f, w - padding, trackY - 6f, trackPaint)
        canvas.drawLine(padding, trackY + 6f, w - padding, trackY + 6f, trackPaint)

        // Draw stations
        val stationNames = listOf("BUFFER", "STFT", "AI", "ISTFT", "OUT")
        for (i in 0..4) {
            val stationX = padding + i * stationSpacing
            val isActive = trainPosition.toInt() == i
            val isPassed = trainPosition > i

            // Station circle
            stationPaint.color = if (isActive || isPassed) stationColors[i] else colorStation
            stationPaint.alpha = if (isActive) 255 else if (isPassed) 180 else 100
            canvas.drawCircle(stationX, stationY, stationRadius, stationPaint)
            stationPaint.alpha = 255

            // Station border (glow when active)
            if (isActive) {
                stationBorderPaint.color = stationColors[i]
                stationBorderPaint.strokeWidth = 3f
                // Pulse effect at output when delivering
                val glowRadius = if (i == 4 && hasCompletedFirstPass) {
                    stationRadius + 4f + kotlin.math.sin(deliveryPulse * Math.PI / 180.0).toFloat() * 3f
                } else {
                    stationRadius + 4f
                }
                canvas.drawCircle(stationX, stationY, glowRadius, stationBorderPaint)
            }

            // Station label
            labelPaint.color = if (isActive) Color.WHITE else Color.parseColor("#888888")
            canvas.drawText(stationNames[i], stationX, stationY + 4f, labelPaint)

            // Connection to track (pole)
            val poleColor = if (isPassed || isActive) stationColors[i] else colorTrack
            trackPaint.color = poleColor
            trackPaint.strokeWidth = 2f
            canvas.drawLine(stationX, stationY + stationRadius, stationX, trackY - 10f, trackPaint)
        }

        // Calculate train X position
        val trainX = padding + trainPosition * stationSpacing

        // Draw smoke puffs (behind train)
        drawSmoke(canvas, trainX, trackY - 25f)

        // Draw train
        drawTrain(canvas, trainX, trackY)

        // Draw status text
        val statusText = getStatusText()
        statusPaint.color = getCurrentColor()
        canvas.drawText(statusText, w / 2, h - 12f, statusPaint)
    }

    private fun drawTrain(canvas: Canvas, x: Float, trackY: Float) {
        val engineWidth = 40f
        val engineHeight = 24f
        val carWidth = 32f
        val carHeight = 20f
        val wheelRadius = 6f

        // Cargo car (behind engine)
        val carX = x - engineWidth/2 - carWidth - 8f
        val carY = trackY - carHeight - wheelRadius

        // Car body
        rect.set(carX, carY, carX + carWidth, carY + carHeight)
        cargoPaint.color = Color.parseColor("#333333")
        canvas.drawRoundRect(rect, 4f, 4f, cargoPaint)

        // Cargo fill (purple audio data)
        if (cargoLevel > 0) {
            val fillHeight = carHeight * cargoLevel * 0.8f
            rect.set(carX + 2f, carY + carHeight - fillHeight - 2f, carX + carWidth - 2f, carY + carHeight - 2f)
            cargoPaint.color = colorCargo
            cargoPaint.alpha = 200
            canvas.drawRoundRect(rect, 2f, 2f, cargoPaint)
            cargoPaint.alpha = 255
        }

        // Car wheels
        drawWheel(canvas, carX + 8f, trackY - wheelRadius, wheelRadius)
        drawWheel(canvas, carX + carWidth - 8f, trackY - wheelRadius, wheelRadius)

        // Coupler
        trainPaint.color = colorWheel
        rect.set(carX + carWidth, trackY - 10f, carX + carWidth + 8f, trackY - 6f)
        canvas.drawRect(rect, trainPaint)

        // Engine body
        val engineX = x - engineWidth/2
        val engineY = trackY - engineHeight - wheelRadius

        // Main body
        trainPaint.color = colorTrain
        rect.set(engineX, engineY + 4f, engineX + engineWidth, engineY + engineHeight)
        canvas.drawRoundRect(rect, 4f, 4f, trainPaint)

        // Cabin
        trainPaint.color = Color.parseColor("#008B9A")
        rect.set(engineX + engineWidth - 16f, engineY, engineX + engineWidth, engineY + engineHeight)
        canvas.drawRoundRect(rect, 4f, 4f, trainPaint)

        // Smokestack
        trainPaint.color = colorTrain
        rect.set(engineX + 6f, engineY - 6f, engineX + 14f, engineY + 4f)
        canvas.drawRoundRect(rect, 2f, 2f, trainPaint)

        // Cow catcher (front)
        val path = Path().apply {
            moveTo(engineX + engineWidth, trackY - wheelRadius)
            lineTo(engineX + engineWidth + 10f, trackY)
            lineTo(engineX + engineWidth, trackY)
            close()
        }
        trainPaint.color = Color.parseColor("#555555")
        canvas.drawPath(path, trainPaint)

        // Engine wheels
        drawWheel(canvas, engineX + 10f, trackY - wheelRadius, wheelRadius)
        drawWheel(canvas, engineX + engineWidth - 10f, trackY - wheelRadius, wheelRadius)
    }

    private fun drawWheel(canvas: Canvas, cx: Float, cy: Float, radius: Float) {
        // Wheel body
        wheelPaint.color = colorWheel
        canvas.drawCircle(cx, cy, radius, wheelPaint)

        // Wheel spokes (rotating)
        wheelPaint.color = Color.parseColor("#555555")
        canvas.save()
        canvas.rotate(wheelRotation, cx, cy)
        for (i in 0..3) {
            canvas.drawLine(cx - radius + 1f, cy, cx + radius - 1f, cy, wheelPaint)
            canvas.rotate(45f, cx, cy)
        }
        canvas.restore()

        // Center hub
        wheelPaint.color = Color.parseColor("#AAAAAA")
        canvas.drawCircle(cx, cy, radius * 0.3f, wheelPaint)
    }

    private fun drawSmoke(canvas: Canvas, x: Float, y: Float) {
        if (currentState == SourceSeparator.SeparationState.IDLE ||
            currentState == SourceSeparator.SeparationState.EXHAUSTED) {
            return
        }

        // Draw smoke puffs drifting up and back
        val numPuffs = 4
        for (i in 0 until numPuffs) {
            val puffAge = (smokeOffset + i * 25f) % 100f
            val puffX = x - 20f - puffAge * 0.5f
            val puffY = y - puffAge * 0.4f
            val puffSize = 4f + puffAge * 0.08f
            val alpha = (255 * (1f - puffAge / 100f)).toInt().coerceIn(0, 255)

            smokePaint.alpha = alpha
            canvas.drawCircle(puffX, puffY, puffSize, smokePaint)
        }
        smokePaint.alpha = 255
    }

    private fun getStatusText(): String {
        // Once playing continuously, show simpler status
        if (hasCompletedFirstPass) {
            return when (currentState) {
                SourceSeparator.SeparationState.PLAYING -> "Streaming audio"
                SourceSeparator.SeparationState.EXHAUSTED -> "Buffer empty..."
                else -> "Processing..."
            }
        }

        // Initial processing - show detailed status
        return when (currentState) {
            SourceSeparator.SeparationState.IDLE -> "Ready"
            SourceSeparator.SeparationState.BUFFERING -> "Buffering ${(progress * 100).toInt()}%"
            SourceSeparator.SeparationState.PROCESSING_STFT -> "Analyzing..."
            SourceSeparator.SeparationState.PROCESSING_NPU -> "AI ${(progress * 100).toInt()}%"
            SourceSeparator.SeparationState.PROCESSING_ISTFT -> "Rebuilding..."
            SourceSeparator.SeparationState.PLAYING -> "Delivering"
            SourceSeparator.SeparationState.EXHAUSTED -> "Waiting..."
        }
    }

    private fun getCurrentColor(): Int {
        return when (currentState) {
            SourceSeparator.SeparationState.IDLE -> Color.parseColor("#666666")
            SourceSeparator.SeparationState.BUFFERING -> stationColors[0]
            SourceSeparator.SeparationState.PROCESSING_STFT -> stationColors[1]
            SourceSeparator.SeparationState.PROCESSING_NPU -> stationColors[2]
            SourceSeparator.SeparationState.PROCESSING_ISTFT -> stationColors[3]
            SourceSeparator.SeparationState.PLAYING -> stationColors[4]
            SourceSeparator.SeparationState.EXHAUSTED -> Color.parseColor("#FF5252")
        }
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val desiredWidth = 400
        val desiredHeight = 140

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
