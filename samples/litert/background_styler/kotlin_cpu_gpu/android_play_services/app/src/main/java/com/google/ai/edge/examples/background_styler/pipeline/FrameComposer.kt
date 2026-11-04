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

import android.graphics.Bitmap
import com.google.mlkit.vision.segmentation.SegmentationMask

/** Blends camera frame, ML Kit foreground mask and LiteRT stylized image into one frame. */
class FrameComposer {

  private var frameBuf = IntArray(0)
  private var styledBuf = IntArray(0)
  private var maskBuf = FloatArray(0)

  /** Composed frame: subject kept + background styled. */
  fun compose(frame: Bitmap, styled: Bitmap, mask: SegmentationMask): Bitmap {
    val w = frame.width
    val h = frame.height
    ensureCapacity(w * h)
    frame.getPixels(frameBuf, 0, w, 0, 0, w, h)
    Bitmap.createScaledBitmap(styled, w, h, true).getPixels(styledBuf, 0, w, 0, 0, w, h)
    readMask(mask, w, h)

    val out = IntArray(w * h)
    for (i in 0 until w * h) {
      val a = smoothstep(maskBuf[i])
      out[i] = blend(frameBuf[i], styledBuf[i], a)
    }
    return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
  }

  /** Debug view of the ML Kit path only: tints the detected subject. */
  fun maskOverlay(frame: Bitmap, mask: SegmentationMask): Bitmap {
    val w = frame.width
    val h = frame.height
    ensureCapacity(w * h)
    frame.getPixels(frameBuf, 0, w, 0, 0, w, h)
    readMask(mask, w, h)

    val out = IntArray(w * h)
    for (i in 0 until w * h) {
      out[i] = blend(MASK_TINT, frameBuf[i], smoothstep(maskBuf[i]) * 0.55f)
    }
    return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
  }

  private fun readMask(mask: SegmentationMask, w: Int, h: Int) {
    require(mask.width == w && mask.height == h) {
      "Mask ${mask.width}x${mask.height} != frame ${w}x$h"
    }
    val buffer = mask.buffer
    buffer.rewind()
    buffer.asFloatBuffer().get(maskBuf, 0, w * h)
  }

  private fun ensureCapacity(n: Int) {
    if (frameBuf.size != n) {
      frameBuf = IntArray(n)
      styledBuf = IntArray(n)
      maskBuf = FloatArray(n)
    }
  }

  private companion object {
    const val MASK_TINT = 0xFF4285F4.toInt()

    /** Sharpens soft mask edges a little so the subject outline is cleaner. */
    fun smoothstep(x: Float): Float {
      val t = ((x - 0.25f) / 0.5f).coerceIn(0f, 1f)
      return t * t * (3f - 2f * t)
    }

    /** Returns `fg * a + bg * (1 - a)` per channel. */
    fun blend(fg: Int, bg: Int, a: Float): Int {
      val ia = 1f - a
      val r = (((fg shr 16) and 0xFF) * a + ((bg shr 16) and 0xFF) * ia).toInt()
      val g = (((fg shr 8) and 0xFF) * a + ((bg shr 8) and 0xFF) * ia).toInt()
      val b = ((fg and 0xFF) * a + (bg and 0xFF) * ia).toInt()
      return (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
  }
}
