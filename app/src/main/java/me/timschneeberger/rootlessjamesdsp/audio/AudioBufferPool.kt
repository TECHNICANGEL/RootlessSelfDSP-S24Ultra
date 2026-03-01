package me.timschneeberger.rootlessjamesdsp.audio

import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger

/**
 * S24 ULTRA RAW POWER - Audio Buffer Pool
 *
 * Zero-allocation buffer management for real-time audio processing
 * Eliminates GC pauses during audio capture/playback
 *
 * Features:
 * - Pre-allocated buffer pools
 * - Lock-free concurrent access
 * - Automatic pool growth
 * - Statistics tracking
 *
 * Usage:
 * ```
 * val pool = AudioBufferPool.getFloatPool(bufferSize)
 * val buffer = pool.acquire()
 * // use buffer...
 * pool.release(buffer)
 * ```
 */
object AudioBufferPool {

    // Pool statistics
    private val totalAllocations = AtomicInteger(0)
    private val poolHits = AtomicInteger(0)
    private val poolMisses = AtomicInteger(0)

    // Float buffer pools by size
    private val floatPools = HashMap<Int, FloatBufferPool>()

    // Short buffer pools by size
    private val shortPools = HashMap<Int, ShortBufferPool>()

    // Byte buffer pools by size
    private val bytePools = HashMap<Int, ByteBufferPool>()

    /**
     * Get or create a FloatArray pool for the specified size
     */
    @Synchronized
    fun getFloatPool(size: Int, initialCapacity: Int = 8): FloatBufferPool {
        return floatPools.getOrPut(size) {
            FloatBufferPool(size, initialCapacity)
        }
    }

    /**
     * Get or create a ShortArray pool for the specified size
     */
    @Synchronized
    fun getShortPool(size: Int, initialCapacity: Int = 8): ShortBufferPool {
        return shortPools.getOrPut(size) {
            ShortBufferPool(size, initialCapacity)
        }
    }

    /**
     * Get or create a ByteArray pool for the specified size
     */
    @Synchronized
    fun getBytePool(size: Int, initialCapacity: Int = 8): ByteBufferPool {
        return bytePools.getOrPut(size) {
            ByteBufferPool(size, initialCapacity)
        }
    }

    /**
     * Get pool statistics
     */
    fun getStats(): PoolStats {
        return PoolStats(
            totalAllocations = totalAllocations.get(),
            poolHits = poolHits.get(),
            poolMisses = poolMisses.get(),
            floatPoolCount = floatPools.size,
            shortPoolCount = shortPools.size,
            bytePoolCount = bytePools.size
        )
    }

    /**
     * Clear all pools and reset statistics
     */
    @Synchronized
    fun clearAll() {
        floatPools.values.forEach { it.clear() }
        shortPools.values.forEach { it.clear() }
        bytePools.values.forEach { it.clear() }
        floatPools.clear()
        shortPools.clear()
        bytePools.clear()
        totalAllocations.set(0)
        poolHits.set(0)
        poolMisses.set(0)
    }

    /**
     * Pre-warm pools with common buffer sizes
     */
    fun preWarm() {
        // FFT sizes
        listOf(1024, 2048, 4096).forEach { size ->
            getFloatPool(size, 8).preAllocate(8)
        }

        // Audio buffer sizes
        listOf(4096, 8192, 16384, 32768).forEach { size ->
            getFloatPool(size, 4).preAllocate(4)
            getShortPool(size, 4).preAllocate(4)
        }

        // Byte buffers
        listOf(16384, 32768, 65536).forEach { size ->
            getBytePool(size, 4).preAllocate(4)
        }
    }

    /**
     * Pool statistics data class
     */
    data class PoolStats(
        val totalAllocations: Int,
        val poolHits: Int,
        val poolMisses: Int,
        val floatPoolCount: Int,
        val shortPoolCount: Int,
        val bytePoolCount: Int
    ) {
        val hitRate: Float
            get() = if (totalAllocations > 0) {
                poolHits.toFloat() / totalAllocations
            } else 0f
    }

    /**
     * FloatArray buffer pool
     */
    class FloatBufferPool(
        private val bufferSize: Int,
        initialCapacity: Int = 8
    ) {
        private val pool = ConcurrentLinkedQueue<FloatArray>()
        private val allocatedCount = AtomicInteger(0)
        private val inUseCount = AtomicInteger(0)

        init {
            preAllocate(initialCapacity)
        }

        /**
         * Pre-allocate buffers to avoid initial GC
         */
        fun preAllocate(count: Int) {
            repeat(count) {
                pool.offer(FloatArray(bufferSize))
                allocatedCount.incrementAndGet()
                totalAllocations.incrementAndGet()
            }
        }

        /**
         * Acquire a buffer from the pool
         * If pool is empty, allocates a new buffer
         */
        fun acquire(): FloatArray {
            val buffer = pool.poll()
            return if (buffer != null) {
                poolHits.incrementAndGet()
                inUseCount.incrementAndGet()
                buffer
            } else {
                poolMisses.incrementAndGet()
                totalAllocations.incrementAndGet()
                allocatedCount.incrementAndGet()
                inUseCount.incrementAndGet()
                FloatArray(bufferSize)
            }
        }

        /**
         * Release a buffer back to the pool
         * Buffer contents are NOT cleared for performance
         */
        fun release(buffer: FloatArray) {
            if (buffer.size == bufferSize) {
                pool.offer(buffer)
                inUseCount.decrementAndGet()
            }
            // If wrong size, let it be GC'd
        }

        /**
         * Release a buffer and zero it (for security-sensitive data)
         */
        fun releaseAndClear(buffer: FloatArray) {
            if (buffer.size == bufferSize) {
                buffer.fill(0f)
                pool.offer(buffer)
                inUseCount.decrementAndGet()
            }
        }

        /**
         * Get number of available buffers in pool
         */
        fun availableCount(): Int = pool.size

        /**
         * Get total allocated buffer count
         */
        fun allocatedCount(): Int = allocatedCount.get()

        /**
         * Get number of buffers currently in use
         */
        fun inUseCount(): Int = inUseCount.get()

        /**
         * Clear the pool
         */
        fun clear() {
            pool.clear()
            allocatedCount.set(0)
            inUseCount.set(0)
        }
    }

    /**
     * ShortArray buffer pool
     */
    class ShortBufferPool(
        private val bufferSize: Int,
        initialCapacity: Int = 8
    ) {
        private val pool = ConcurrentLinkedQueue<ShortArray>()
        private val allocatedCount = AtomicInteger(0)
        private val inUseCount = AtomicInteger(0)

        init {
            preAllocate(initialCapacity)
        }

        fun preAllocate(count: Int) {
            repeat(count) {
                pool.offer(ShortArray(bufferSize))
                allocatedCount.incrementAndGet()
                totalAllocations.incrementAndGet()
            }
        }

        fun acquire(): ShortArray {
            val buffer = pool.poll()
            return if (buffer != null) {
                poolHits.incrementAndGet()
                inUseCount.incrementAndGet()
                buffer
            } else {
                poolMisses.incrementAndGet()
                totalAllocations.incrementAndGet()
                allocatedCount.incrementAndGet()
                inUseCount.incrementAndGet()
                ShortArray(bufferSize)
            }
        }

        fun release(buffer: ShortArray) {
            if (buffer.size == bufferSize) {
                pool.offer(buffer)
                inUseCount.decrementAndGet()
            }
        }

        fun availableCount(): Int = pool.size
        fun allocatedCount(): Int = allocatedCount.get()
        fun inUseCount(): Int = inUseCount.get()

        fun clear() {
            pool.clear()
            allocatedCount.set(0)
            inUseCount.set(0)
        }
    }

    /**
     * ByteArray buffer pool
     */
    class ByteBufferPool(
        private val bufferSize: Int,
        initialCapacity: Int = 8
    ) {
        private val pool = ConcurrentLinkedQueue<ByteArray>()
        private val allocatedCount = AtomicInteger(0)
        private val inUseCount = AtomicInteger(0)

        init {
            preAllocate(initialCapacity)
        }

        fun preAllocate(count: Int) {
            repeat(count) {
                pool.offer(ByteArray(bufferSize))
                allocatedCount.incrementAndGet()
                totalAllocations.incrementAndGet()
            }
        }

        fun acquire(): ByteArray {
            val buffer = pool.poll()
            return if (buffer != null) {
                poolHits.incrementAndGet()
                inUseCount.incrementAndGet()
                buffer
            } else {
                poolMisses.incrementAndGet()
                totalAllocations.incrementAndGet()
                allocatedCount.incrementAndGet()
                inUseCount.incrementAndGet()
                ByteArray(bufferSize)
            }
        }

        fun release(buffer: ByteArray) {
            if (buffer.size == bufferSize) {
                pool.offer(buffer)
                inUseCount.decrementAndGet()
            }
        }

        fun availableCount(): Int = pool.size
        fun allocatedCount(): Int = allocatedCount.get()
        fun inUseCount(): Int = inUseCount.get()

        fun clear() {
            pool.clear()
            allocatedCount.set(0)
            inUseCount.set(0)
        }
    }
}
