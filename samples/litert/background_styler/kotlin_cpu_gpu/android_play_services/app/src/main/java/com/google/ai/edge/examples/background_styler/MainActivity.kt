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

package com.google.ai.edge.examples.background_styler

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.ai.edge.examples.background_styler.pipeline.StylerViewModel
import com.google.ai.edge.examples.background_styler.pipeline.UiState
import com.google.ai.edge.examples.background_styler.pipeline.ViewMode
import com.google.ai.edge.litert.Accelerator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContent { MaterialTheme(colorScheme = darkColorScheme()) { App() } }
  }
}

@Composable
private fun App() {
  val context = LocalContext.current
  var hasPermission by remember {
    mutableStateOf(
      ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED
    )
  }
  val launcher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
      hasPermission = it
    }
  LaunchedEffect(Unit) { if (!hasPermission) launcher.launch(Manifest.permission.CAMERA) }

  if (hasPermission) {
    StylerScreen()
  } else {
    Box(Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
      Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) {
        Text("Grant camera permission")
      }
    }
  }
}

@Composable
private fun StylerScreen(vm: StylerViewModel = viewModel()) {
  val state by vm.state.collectAsStateWithLifecycle()
  var useFrontCamera by rememberSaveable { mutableStateOf(true) }

  CameraFeed(vm, useFrontCamera)

  Box(Modifier.fillMaxSize().background(Color.Black)) {
    state.frame?.let {
      Image(
        bitmap = it.asImageBitmap(),
        contentDescription = "Processed camera frame",
        contentScale = ContentScale.Crop,
        modifier = Modifier.fillMaxSize(),
      )
    }

    StatsPanel(state, Modifier.align(Alignment.TopStart).statusBarsPadding().padding(12.dp))

    if (state.modelsLoading) {
      Column(
        Modifier.align(Alignment.Center),
        horizontalAlignment = Alignment.CenterHorizontally,
      ) {
        CircularProgressIndicator()
        Spacer(Modifier.size(8.dp))
        Text("Compiling LiteRT model for ${state.requestedAccelerator}…", color = Color.White)
      }
    }

    ControlsPanel(
      state = state,
      vm = vm,
      onFlipCamera = { useFrontCamera = !useFrontCamera },
      modifier = Modifier.align(Alignment.BottomCenter).navigationBarsPadding(),
    )
  }
}

/** Binds a CameraX ImageAnalysis use case that feeds frames into the ML pipeline. */
@Composable
private fun CameraFeed(vm: StylerViewModel, useFrontCamera: Boolean) {
  val context = LocalContext.current
  val lifecycleOwner = LocalLifecycleOwner.current
  LaunchedEffect(useFrontCamera) {
    val provider =
      withContext(Dispatchers.IO) { ProcessCameraProvider.getInstance(context).get() }
    // Not every device has both cameras, so use the one it has.
    val useFront =
      if (useFrontCamera) provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)
      else !provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)
    val analysis =
      ImageAnalysis.Builder()
        .setResolutionSelector(
          ResolutionSelector.Builder()
            .setResolutionStrategy(
              ResolutionStrategy(
                Size(640, 480),
                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
              )
            )
            .build()
        )
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .build()
        .also { it.setAnalyzer(vm.mlExecutor) { image -> vm.analyze(image, useFront) } }
    val selector = if (useFront) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
    provider.unbindAll()
    provider.bindToLifecycle(lifecycleOwner, selector, analysis)
  }
}

@Composable
private fun StatsPanel(state: UiState, modifier: Modifier = Modifier) {
  Column(
    modifier
      .clip(RoundedCornerShape(12.dp))
      .background(Color.Black.copy(alpha = 0.6f))
      .padding(horizontal = 12.dp, vertical = 8.dp)
  ) {
    // Pre/post-processing and compose timings are still measured and logged under the
    // `PipelineCheck` logcat tag; the on-screen overlay shows only the headline numbers.
    val mono = FontFamily.Monospace
    StatLine("ML Kit Selfie Segmentation (CPU)", "%.1f ms".format(state.mlKitMs), Color(0xFFFFEB3B), mono)
    val accel = state.activeAccelerator?.name ?: "…"
    StatLine("LiteRT inference ($accel)", "%.1f ms".format(state.liteRtInferenceMs), Color(0xFFFFEB3B), mono)
    StatLine("FPS", "%.1f".format(state.fps), Color.White, mono)
    if (state.gpuFallback) {
      Text("GPU not available, running on CPU", color = Color(0xFFF28B82), fontSize = 11.sp)
    }
    // Only shown if loading a model or changing the style fails.
    state.error?.let { Text(it, color = Color(0xFFF28B82), fontSize = 11.sp) }
  }
}

@Composable
private fun StatLine(label: String, value: String, labelColor: Color, mono: FontFamily) {
  Row {
    Text(label, color = labelColor, fontSize = 12.sp, modifier = Modifier.width(230.dp))
    Text(value, color = Color.White, fontSize = 12.sp, fontFamily = mono)
  }
}

@Composable
private fun ControlsPanel(
  state: UiState,
  vm: StylerViewModel,
  onFlipCamera: () -> Unit,
  modifier: Modifier = Modifier,
) {
  Column(
    modifier
      .fillMaxWidth()
      .background(Color.Black.copy(alpha = 0.65f))
      .padding(12.dp),
    verticalArrangement = Arrangement.spacedBy(10.dp),
  ) {
    // View mode: full pipeline or either path in isolation.
    val modes = ViewMode.entries
    SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
      modes.forEachIndexed { i, m ->
        SegmentedButton(
          selected = state.mode == m,
          onClick = { vm.setMode(m) },
          shape = SegmentedButtonDefaults.itemShape(i, modes.size),
          label = { Text(m.label, fontSize = 11.sp, maxLines = 1) },
        )
      }
    }

    // Style picker -> re-runs the style prediction model once.
    LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
      itemsIndexed(vm.styles) { i, style ->
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
          Image(
            bitmap = style.thumbnail.asImageBitmap(),
            contentDescription = style.name,
            contentScale = ContentScale.Crop,
            modifier =
              Modifier.size(64.dp)
                .clip(RoundedCornerShape(10.dp))
                .border(
                  BorderStroke(
                    3.dp,
                    if (i == state.selectedStyle) Color(0xFF8AB4F8) else Color.Transparent,
                  ),
                  RoundedCornerShape(10.dp),
                )
                .clickable { vm.selectStyle(i) },
          )
          Text(style.name, color = Color.White, fontSize = 11.sp)
        }
      }
    }

    Row(verticalAlignment = Alignment.CenterVertically) {
      Text("LiteRT accelerator:", color = Color.White, fontSize = 13.sp)
      Spacer(Modifier.width(8.dp))
      listOf(Accelerator.GPU, Accelerator.CPU).forEach { a ->
        FilterChip(
          selected = state.requestedAccelerator == a,
          onClick = { vm.setAccelerator(a) },
          enabled = !state.modelsLoading,
          label = { Text(a.name) },
          modifier = Modifier.padding(end = 8.dp),
        )
      }
      Spacer(Modifier.weight(1f))
      IconButton(onClick = onFlipCamera) {
        Icon(
          painterResource(R.drawable.ic_cameraswitch),
          contentDescription = "Switch camera",
          tint = Color.White,
        )
      }
    }
  }
}
