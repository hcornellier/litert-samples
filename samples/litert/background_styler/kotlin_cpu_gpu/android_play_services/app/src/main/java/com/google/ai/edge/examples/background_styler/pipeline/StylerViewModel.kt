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

package com.google.ai.edge.examples.background_styler.pipeline

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import com.google.ai.edge.examples.background_styler.litert.LiteRtStylizer
import com.google.ai.edge.examples.background_styler.mlkit.SelfieSegmenterHelper
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.Environment
import com.google.android.gms.litert.LiteRt
import com.google.android.gms.litert.LiteRtDsl.initializationOptions
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.segmentation.SegmentationMask
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

/** What the viewfinder shows. [COMPOSED] is the full two-path pipeline. */
enum class ViewMode(val label: String) {
  COMPOSED("Composed"),
  MASK("ML Kit mask"),
  STYLIZED("LiteRT style"),
  ORIGINAL("Original"),
}

data class StyleOption(val name: String, val thumbnail: Bitmap)

data class UiState(
  val frame: Bitmap? = null,
  val mode: ViewMode = ViewMode.COMPOSED,
  val selectedStyle: Int = 0,
  val requestedAccelerator: Accelerator = Accelerator.GPU,
  val activeAccelerator: Accelerator? = null,
  val modelsLoading: Boolean = true,
  /** ML Kit Selfie Segmentation end-to-end latency (runs on CPU). */
  val mlKitMs: Float = 0f,
  /** LiteRT CompiledModel.run() + output readback -- inference on the active accelerator. */
  val liteRtInferenceMs: Float = 0f,
  val fps: Float = 0f,
  val error: String? = null,
) {
  /** GPU was requested but LiteRT had to fall back to CPU. */
  val gpuFallback: Boolean
    get() = requestedAccelerator == Accelerator.GPU && activeAccelerator == Accelerator.CPU
}

/**
 * One app, two paths: every camera frame goes to
 * 1. ML Kit Selfie Segmentation -> foreground mask, and
 * 2. a custom LiteRT CompiledModel on GPU or CPU -> stylized image,
 * which are then composed into "subject kept + background styled".
 */
class StylerViewModel(private val app: Application) : AndroidViewModel(app) {

  /**
   * All ML work (model compile, style prediction, per-frame inference) runs on this one thread.
   * CameraX delivers frames on it too, so GPU objects are always used from the thread that
   * created them, and frames are naturally dropped while one is being processed.
   */
  val mlExecutor: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "ml") }

  /**
   * Runs every LiteRT call (Play services init, compile, style prediction, close) on [mlExecutor].
   * This is the ViewModel's own scope, not `viewModelScope`: `viewModelScope` is canceled before
   * [onCleared] runs, so a close launched there would not run.
   */
  private val mlScope = CoroutineScope(SupervisorJob() + mlExecutor.asCoroutineDispatcher())

  /**
   * Waits on ML Kit results and timestamps them. Timing ML Kit here (instead of via a completion
   * listener on the ML thread) avoids a race where Tasks.await() returned before the listener ran
   * and ML Kit was reported with LiteRT's latency.
   */
  private val mlKitTimer: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "mlkit-timer") }

  private val segmenter = SelfieSegmenterHelper()
  private val stylizer = LiteRtStylizer(app)
  private val composer = FrameComposer()

  /** LiteRT environment from Google Play services; created once, reused across recompiles. */
  private var environment: Environment? = null

  val styles: List<StyleOption> =
    app.assets.list(STYLE_DIR).orEmpty().sorted().map { file ->
      val bmp = app.assets.open("$STYLE_DIR/$file").use { BitmapFactory.decodeStream(it) }
      StyleOption(file.substringBeforeLast('.').replaceFirstChar { it.uppercase() }, bmp)
    }

  private val _state = MutableStateFlow(UiState())
  val state: StateFlow<UiState> = _state.asStateFlow()

  @Volatile private var mode = ViewMode.COMPOSED
  private var lastFrameTime = 0L
  private var frameCount = 0L
  private var mlKitLogged = false

  init {
    mlScope.launch {
      try {
        // LiteRT runtime comes from Google Play services (no LiteRT native libs in the APK).
        environment = LiteRt.initialize(app, initializationOptions { setEnableGpu(true) }).await()
        Log.i(CHECK, "LiteRT in Google Play services initialized (GPU support requested)")
        loadModels(Accelerator.GPU)
      } catch (e: Exception) {
        Log.e(CHECK, "LiteRT in Google Play services init failed", e)
        _state.update {
          it.copy(modelsLoading = false, error = "Play services LiteRT init failed: ${e.message}")
        }
      }
    }
  }

  fun setMode(newMode: ViewMode) {
    mode = newMode
    _state.update { it.copy(mode = newMode) }
  }

  fun setAccelerator(accelerator: Accelerator) {
    if (accelerator == _state.value.requestedAccelerator) return
    mlScope.launch { loadModels(accelerator) }
  }

  fun selectStyle(index: Int) {
    _state.update { it.copy(selectedStyle = index) }
    mlScope.launch {
      // Before the models are loaded there is nothing to run: loadModels() applies the selection.
      if (stylizer.activeAccelerator == null) return@launch
      try {
        stylizer.setStyle(styles[index].thumbnail)
      } catch (e: Exception) {
        Log.e(TAG, "Style change failed", e)
        _state.update { it.copy(error = "Style change failed: ${e.message}") }
      }
    }
  }

  /** Runs on [mlScope]. */
  private fun loadModels(accelerator: Accelerator) {
    _state.update {
      it.copy(requestedAccelerator = accelerator, modelsLoading = true, error = null)
    }
    try {
      stylizer.initialize(accelerator, checkNotNull(environment) { "LiteRT not initialized" })
      stylizer.setStyle(styles[_state.value.selectedStyle].thumbnail)
      val active = stylizer.activeAccelerator
      _state.update { it.copy(activeAccelerator = active, modelsLoading = false) }
      if (active == accelerator) {
        Log.i(CHECK, "LiteRT CompiledModel style model compiled for $active")
      } else {
        Log.w(CHECK, "LiteRT requested $accelerator but is running on $active (fallback)")
      }
    } catch (e: Exception) {
      Log.e(TAG, "Model load failed", e)
      _state.update { it.copy(modelsLoading = false, error = "Model load failed: ${e.message}") }
    }
  }

  /** CameraX analyzer callback; runs on [mlExecutor]. */
  fun analyze(image: ImageProxy, isFrontCamera: Boolean) {
    val frame =
      try {
        uprightBitmap(image, isFrontCamera)
      } finally {
        image.close()
      }

    val currentMode = mode
    var mlKitMs = 0f
    var liteRtInferenceMs = 0f
    var liteRtPrePostMs = 0f
    var composeMs = 0f
    val output: Bitmap =
      try {
        when (currentMode) {
          ViewMode.ORIGINAL -> frame
          ViewMode.MASK -> {
            val (mask, ms) = segmentTimed(frame)
            mlKitMs = ms
            onMaskReceived(mask.width, mask.height)
            composer.maskOverlay(frame, mask)
          }
          ViewMode.STYLIZED -> {
            if (!stylizer.isReady) frame
            else {
              val styled = stylizer.stylize(frame)
              stylizer.lastTimings.let {
                liteRtInferenceMs = it.inferenceMs
                liteRtPrePostMs = it.preMs + it.postMs
              }
              Bitmap.createScaledBitmap(styled, frame.width, frame.height, true)
            }
          }
          ViewMode.COMPOSED -> {
            if (!stylizer.isReady) frame
            else {
              // Both paths run concurrently: ML Kit (CPU) is timed on its own thread while
              // LiteRT runs on this one.
              val maskFuture = mlKitTimer.submit<Pair<SegmentationMask, Float>> { segmentTimed(frame) }
              val styled = stylizer.stylize(frame)
              stylizer.lastTimings.let {
                liteRtInferenceMs = it.inferenceMs
                liteRtPrePostMs = it.preMs + it.postMs
              }
              val (mask, ms) = maskFuture.get()
              mlKitMs = ms
              onMaskReceived(mask.width, mask.height)
              val t1 = System.nanoTime()
              val composed = composer.compose(frame, styled, mask)
              composeMs = (System.nanoTime() - t1) / 1_000_000f
              composed
            }
          }
        }
      } catch (e: Exception) {
        Log.e(TAG, "Frame processing failed", e)
        frame
      }

    if (++frameCount % 60 == 0L && currentMode == ViewMode.COMPOSED && stylizer.isReady) {
      Log.i(
        CHECK,
        "frame=$frameCount mlkit(CPU)=%.1fms litert-inference(%s)=%.1fms litert-pre/post(CPU)=%.1fms compose=%.1fms"
          .format(mlKitMs, stylizer.activeAccelerator, liteRtInferenceMs, liteRtPrePostMs, composeMs),
      )
    }

    val now = SystemClock.uptimeMillis()
    val instFps = if (lastFrameTime > 0) 1000f / (now - lastFrameTime).coerceAtLeast(1) else 0f
    lastFrameTime = now
    _state.update {
      it.copy(
        frame = output,
        mlKitMs = mlKitMs,
        liteRtInferenceMs = liteRtInferenceMs,
        fps = if (it.fps == 0f) instFps else it.fps * 0.9f + instFps * 0.1f,
      )
    }
  }

  /** Runs ML Kit segmentation and returns the mask with its end-to-end latency in ms. */
  private fun segmentTimed(frame: Bitmap): Pair<SegmentationMask, Float> {
    val start = System.nanoTime()
    val mask = Tasks.await(segmenter.process(frame))
    return mask to (System.nanoTime() - start) / 1_000_000f
  }

  private fun onMaskReceived(width: Int, height: Int) {
    if (mlKitLogged) return
    mlKitLogged = true
    Log.i(CHECK, "ML Kit Selfie Segmentation (STREAM_MODE) returned mask ${width}x$height")
  }

  private fun uprightBitmap(image: ImageProxy, mirror: Boolean): Bitmap {
    val src = image.toBitmap()
    val rotation = image.imageInfo.rotationDegrees
    if (rotation == 0 && !mirror) return src
    val matrix =
      Matrix().apply {
        postRotate(rotation.toFloat())
        if (mirror) postScale(-1f, 1f)
      }
    return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
  }

  override fun onCleared() {
    mlScope
      .launch {
        stylizer.close()
        segmenter.close()
        environment?.close()
        environment = null
      }
      .invokeOnCompletion { mlScope.cancel() }
    mlExecutor.shutdown()
    mlKitTimer.shutdown()
  }

  private companion object {
    const val TAG = "StylerViewModel"
    /** `adb logcat -s PipelineCheck` shows which ML paths are live and on which accelerator. */
    const val CHECK = "PipelineCheck"
    const val STYLE_DIR = "styles"
  }
}
