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

package com.google.ai.edge.examples.background_styler.mlkit

import android.graphics.Bitmap
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.SegmentationMask
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions

/**
 * ML Kit Selfie Segmentation.
 *
 * ML Kit APIs are regular app-level APIs: images in, [SegmentationMask] out -- no tensors.
 */
class SelfieSegmenterHelper : AutoCloseable {

  // 1. Configure and create the segmenter client (STREAM_MODE = optimized for live camera).
  private val segmenter =
    Segmentation.getClient(
      SelfieSegmenterOptions.Builder()
        .setDetectorMode(SelfieSegmenterOptions.STREAM_MODE)
        .build()
    )

  /**
   * 2. Prepare the image and run inference. The frame is already upright, so rotation is 0.
   *
   * The returned mask has the same size as [frame]; each value is the foreground confidence in
   * [0, 1].
   */
  fun process(frame: Bitmap): Task<SegmentationMask> =
    segmenter.process(InputImage.fromBitmap(frame, /* rotationDegrees= */ 0))

  override fun close() = segmenter.close()
}
