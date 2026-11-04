# Background Styler with ML Kit and LiteRT in Google Play services

This sample uses ML Kit and LiteRT together in one Android app, with the LiteRT
runtime coming from **Google Play services**. It keeps the person in the camera
frame and restyles the background in real time:

```
Camera ──frame──► ML Kit Selfie Segmentation ──foreground mask──┐
   │                                                            ▼
   ├──frame──► LiteRT CompiledModel (GPU or CPU) ──stylized──► Composed frame
   │                                                            ▲
   └──frame─────────────────────────────────────────────────────┘
```

| Path | Code | Model |
|---|---|---|
| ML Kit Selfie Segmentation (`STREAM_MODE`, runs on the CPU) | `mlkit/SelfieSegmenterHelper.kt` | Included in `com.google.mlkit:segmentation-selfie` |
| LiteRT `CompiledModel` on the GPU or CPU | `litert/LiteRtStylizer.kt` | Magenta arbitrary image stylization v1-256, fp16 (`style_predict_fp16.tflite`, `style_transfer_fp16.tflite`) |
| Compositing | `pipeline/FrameComposer.kt` | |
| Orchestration and timing | `pipeline/StylerViewModel.kt` | |

Each camera frame goes to both paths at the same time, so a frame costs about
the slower of the two plus compositing, not their sum.

## LiteRT in Google Play services

The app depends on `com.google.android.gms:play-services-litert`, so the LiteRT
runtime and the GPU accelerator are not packaged in the APK. Before creating a
`CompiledModel`, it gets a LiteRT `Environment` from Google Play services and
passes it to `CompiledModel.create(...)`:

```kotlin
val environment =
  LiteRt.initialize(context, initializationOptions { setEnableGpu(true) }).await()
val model =
  CompiledModel.create(
    context.assets,
    "style_transfer_fp16.tflite",
    CompiledModel.Options(Accelerator.GPU),
    environment,
  )
```

See `StylerViewModel.kt` and `LiteRtStylizer.kt`. The environment is created
once and shared by every `CompiledModel`, and it is closed when the `ViewModel`
is cleared.

This sample requests `Accelerator.GPU` alone so that the GPU / CPU toggle shows
which accelerator the model really runs on. If either model cannot run on the
GPU, the app reloads both on the CPU and says so on screen. If you don't need
that, use `CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)` as described
in the
[image segmentation sample](https://github.com/google-ai-edge/litert-samples/tree/main/samples/litert/image_segmentation/kotlin_cpu_gpu/android_play_services).

`play-services-litert` is built with Kotlin 2.4.0, so the sample uses Kotlin
2.4.0 (see `gradle/libs.versions.toml`). If you enable minification in your own
app, use R8 9.1 or newer.

## What the app shows

*   **Stats:** ML Kit latency, LiteRT inference time on the active accelerator
    (`CompiledModel.run()` plus reading the output, because on the GPU `run()`
    only enqueues the work), and the frame rate of the whole pipeline.
*   **Mode:** `Composed` (full pipeline), `ML Kit mask`, `LiteRT style` or
    `Original`.
*   **Style picker:** picking one of the four style images runs the style
    prediction model once.
*   **GPU / CPU toggle:** recompiles the models on the chosen accelerator.
*   **Camera flip:** switches between the front and back cameras.

More timings, including pre- and post-processing and compositing, are logged
every 60 frames in `Composed` mode:

```sh
adb logcat -s PipelineCheck
```

## Requirements

*   An arm64 Android device or emulator (API level 26 or higher) with a camera
    and an up-to-date version of Google Play services. A physical device is
    recommended to try the GPU.
*   Android Studio, or the Android SDK and JDK 17 to build from the command
    line.

## Build and run

To build and install a debug version:

```sh
./gradlew :app:installDebug
```

To build and install an optimized release version:

```sh
./gradlew :app:installRelease
```

Use the release version for any size or performance measurements; the debug
version is not optimized. For convenience, the sample signs release builds with
the debug key; your own app must use its own release signing key.

The style models are downloaded into `app/src/main/assets` during the build (see
`app/download_model.gradle`).

In some cases, Google Play services may need to download the LiteRT module the
first time the app uses LiteRT, so the first initialization can take longer, or
fail if the device is offline. It also fails if the device's Google Play
services does not have the LiteRT module. In either case the app shows an error.

The GPU accelerator of LiteRT in Google Play services may not be available on
every device yet, and it may only become available after the app's first launch.
Until then the app runs on the CPU and says so on screen. Restart the app to try
the GPU again.

## Assets

*   The style models are the fp16 versions of Magenta's arbitrary image
    stylization v1-256 models (Apache 2.0).
*   The style images in `assets/styles/` are AI-generated for this sample.
