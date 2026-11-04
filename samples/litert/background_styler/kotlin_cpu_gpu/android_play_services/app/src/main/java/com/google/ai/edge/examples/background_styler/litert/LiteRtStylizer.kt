/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.background_styler.litert

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer

/**
 * A custom `.tflite` model run with the LiteRT CompiledModel API on GPU or CPU.
 *
 * Uses the open-source Magenta arbitrary-image-stylization-v1-256 (fp16) models:
 * - `style_predict_fp16.tflite`: style image [1,256,256,3] -> style bottleneck [1,1,1,100]. Run once
 *   per style selection.
 * - `style_transfer_fp16.tflite`: content [1,384,384,3] + bottleneck [1,1,1,100] -> stylized image
 *   [1,384,384,3] in [0,1]. Run every camera frame.
 *
 * Not thread-safe: create, use and close from a single thread.
 */
class LiteRtStylizer(private val context: Context) : AutoCloseable {

  private var predictModel: CompiledModel? = null
  private var transferModel: CompiledModel? = null

  // Allocated once per load and reused -- no per-frame allocations on the inference path.
  private var predictInputs: List<TensorBuffer> = emptyList()
  private var predictOutputs: List<TensorBuffer> = emptyList()
  private var transferInputs: List<TensorBuffer> = emptyList()
  private var transferOutputs: List<TensorBuffer> = emptyList()

  private val contentPixels = IntArray(CONTENT_SIZE * CONTENT_SIZE)
  private val contentFloats = FloatArray(CONTENT_SIZE * CONTENT_SIZE * 3)
  private val outputPixels = IntArray(CONTENT_SIZE * CONTENT_SIZE)

  private var styleBottleneck: FloatArray? = null

  /** The accelerator actually in use (may differ from the requested one after fallback). */
  var activeAccelerator: Accelerator? = null
    private set

  val isReady: Boolean
    get() = transferModel != null && styleBottleneck != null

  /**
   * Compiles both models for [preferred], falling back to CPU if the GPU is unavailable.
   *
   * Call off the main thread: accelerator probing and GPU shader compilation take time. If loading
   * fails on every accelerator, all models and buffers are closed before the exception is rethrown.
   *
   * @param environment LiteRT environment backed by the Google Play services runtime (from
   *   `LiteRt.initialize(context, options)`).
   */
  fun initialize(preferred: Accelerator, environment: Environment) {
    closeModels()
    activeAccelerator =
      try {
        loadModels(preferred, environment)
        preferred
      } catch (e: Exception) {
        closeModels()
        if (preferred == Accelerator.CPU) throw e
        Log.w(TAG, "Failed to compile for $preferred, falling back to CPU", e)
        try {
          loadModels(Accelerator.CPU, environment)
        } catch (cpuError: Exception) {
          closeModels()
          throw cpuError
        }
        Accelerator.CPU
      }
    Log.i(TAG, "Stylizer ready on $activeAccelerator")
  }

  private fun loadModels(accelerator: Accelerator, environment: Environment) {
    val predict =
      CompiledModel.create(
        context.assets,
        PREDICT_MODEL,
        CompiledModel.Options(accelerator),
        environment,
      )
    predictModel = predict
    predictInputs = predict.createInputBuffers()
    predictOutputs = predict.createOutputBuffers()
    val transfer =
      CompiledModel.create(
        context.assets,
        TRANSFER_MODEL,
        CompiledModel.Options(accelerator),
        environment,
      )
    transferModel = transfer
    transferInputs = transfer.createInputBuffers()
    transferOutputs = transfer.createOutputBuffers()

    // Warm-up: the first run on the GPU compiles shaders. Doing it here keeps that cost out of the
    // first camera frame's timing, and a GPU failure at run time triggers the CPU fallback in
    // initialize().
    predictInputs[0].writeFloat(FloatArray(STYLE_SIZE * STYLE_SIZE * 3))
    predict.run(predictInputs, predictOutputs)
    predictOutputs[0].readFloat()
    transferInputs[CONTENT_INPUT].writeFloat(FloatArray(CONTENT_SIZE * CONTENT_SIZE * 3))
    transferInputs[BOTTLENECK_INPUT].writeFloat(FloatArray(BOTTLENECK_SIZE))
    transfer.run(transferInputs, transferOutputs)
    transferOutputs[0].readFloat()
  }

  /** Computes the style bottleneck for [styleImage]. Call when the user picks a new style. */
  fun setStyle(styleImage: Bitmap) {
    val model = checkNotNull(predictModel) { "initialize() first" }
    val scaled = Bitmap.createScaledBitmap(styleImage, STYLE_SIZE, STYLE_SIZE, true)
    val pixels = IntArray(STYLE_SIZE * STYLE_SIZE)
    val floats = FloatArray(STYLE_SIZE * STYLE_SIZE * 3)
    scaled.getPixels(pixels, 0, STYLE_SIZE, 0, 0, STYLE_SIZE, STYLE_SIZE)
    pixelsToRgbFloats(pixels, floats)

    predictInputs[0].writeFloat(floats)
    model.run(predictInputs, predictOutputs)
    styleBottleneck = predictOutputs[0].readFloat()
  }

  /**
   * Per-stage timings of the last [stylize] call, in milliseconds. [inferenceMs] covers
   * `CompiledModel.run()` plus reading the output tensor: on the GPU, `run()` only enqueues the work
   * and the readback waits for it to finish, so timing `run()` alone would leave out the inference.
   */
  data class Timings(val preMs: Float = 0f, val inferenceMs: Float = 0f, val postMs: Float = 0f)

  var lastTimings = Timings()
    private set

  /**
   * Stylizes [frame] and returns a [CONTENT_SIZE]x[CONTENT_SIZE] bitmap. The caller scales it to
   * the frame size when compositing.
   */
  fun stylize(frame: Bitmap): Bitmap {
    check(isReady) { "initialize() and setStyle() first" }
    val model = transferModel!!

    // Pre-process (CPU): resize to 384x384, RGB float in [0, 1], NHWC, write input tensors.
    val t0 = System.nanoTime()
    val scaled = Bitmap.createScaledBitmap(frame, CONTENT_SIZE, CONTENT_SIZE, true)
    scaled.getPixels(contentPixels, 0, CONTENT_SIZE, 0, 0, CONTENT_SIZE, CONTENT_SIZE)
    pixelsToRgbFloats(contentPixels, contentFloats)
    transferInputs[CONTENT_INPUT].writeFloat(contentFloats)
    transferInputs[BOTTLENECK_INPUT].writeFloat(styleBottleneck!!)

    // Inference on the active accelerator, timed together with the output readback.
    val t1 = System.nanoTime()
    model.run(transferInputs, transferOutputs)
    val out = transferOutputs[0].readFloat()

    // Post-process (CPU): [0, 1] RGB floats -> ARGB pixels.
    val t2 = System.nanoTime()
    for (i in outputPixels.indices) {
      val r = (out[i * 3] * 255f).toInt().coerceIn(0, 255)
      val g = (out[i * 3 + 1] * 255f).toInt().coerceIn(0, 255)
      val b = (out[i * 3 + 2] * 255f).toInt().coerceIn(0, 255)
      outputPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    val result =
      Bitmap.createBitmap(outputPixels, CONTENT_SIZE, CONTENT_SIZE, Bitmap.Config.ARGB_8888)
    val t3 = System.nanoTime()

    lastTimings = Timings(preMs = ms(t0, t1), inferenceMs = ms(t1, t2), postMs = ms(t2, t3))
    return result
  }

  private fun ms(startNs: Long, endNs: Long): Float = (endNs - startNs) / 1_000_000f

  private fun pixelsToRgbFloats(pixels: IntArray, out: FloatArray) {
    for (i in pixels.indices) {
      val p = pixels[i]
      out[i * 3] = ((p shr 16) and 0xFF) / 255f
      out[i * 3 + 1] = ((p shr 8) and 0xFF) / 255f
      out[i * 3 + 2] = (p and 0xFF) / 255f
    }
  }

  private fun closeModels() {
    (predictInputs + predictOutputs + transferInputs + transferOutputs).forEach { it.close() }
    predictInputs = emptyList()
    predictOutputs = emptyList()
    transferInputs = emptyList()
    transferOutputs = emptyList()
    transferModel?.close()
    predictModel?.close()
    transferModel = null
    predictModel = null
    activeAccelerator = null
  }

  override fun close() {
    closeModels()
    styleBottleneck = null
  }

  companion object {
    private const val TAG = "LiteRtStylizer"
    const val PREDICT_MODEL = "style_predict_fp16.tflite"
    const val TRANSFER_MODEL = "style_transfer_fp16.tflite"
    const val STYLE_SIZE = 256
    const val CONTENT_SIZE = 384
    // Size of the style bottleneck: style_predict output [1,1,1,100].
    private const val BOTTLENECK_SIZE = 100
    // Input order of style_transfer_fp16.tflite: [content_image, mobilenet_conv/Conv/BiasAdd].
    private const val CONTENT_INPUT = 0
    private const val BOTTLENECK_INPUT = 1
  }
}
