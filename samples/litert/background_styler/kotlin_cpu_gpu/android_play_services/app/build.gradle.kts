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

// Background styler sample: ML Kit and LiteRT in one app, on the LiteRT runtime provided by Google
// Play services.
//
// Example:
//   ./gradlew build

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.undercouch.download)
  alias(libs.plugins.compose.compiler)
}

android {
  namespace = "com.google.ai.edge.examples.background_styler"
  compileSdk = 36

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.background_styler.playservices"
    minSdk = 26
    targetSdk = 36
    versionCode = 1
    versionName = "1.0"

    // Keeps the APK small: ML Kit adds about 10 MB of native code per ABI.
    ndk { abiFilters += "arm64-v8a" }
  }

  buildTypes {
    release {
      // Sign release builds with the debug key so that `./gradlew :app:installRelease` works out of
      // the box, which makes it easy to measure the optimized build. Real apps must sign release
      // builds with their own key.
      signingConfig = signingConfigs.getByName("debug")
      isMinifyEnabled = false
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  buildFeatures { compose = true }

  packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

// Import the model download tasks.
project.extensions.extraProperties["ASSET_DIR"] = "$projectDir/src/main/assets"
apply(from = "download_model.gradle")

dependencies {
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  implementation(libs.androidx.activity.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.graphics)
  implementation(libs.androidx.compose.material3)

  implementation(libs.androidx.camera.core)
  implementation(libs.androidx.camera.camera2)
  implementation(libs.androidx.camera.lifecycle)

  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.kotlinx.coroutines.play.services)

  // ML use #1: ML Kit Selfie Segmentation -> foreground mask.
  implementation(libs.mlkit.segmentation.selfie)

  // ML use #2: LiteRT CompiledModel API -> stylized background, backed by the runtime in Google
  // Play services.
  implementation(libs.play.services.litert)
}
