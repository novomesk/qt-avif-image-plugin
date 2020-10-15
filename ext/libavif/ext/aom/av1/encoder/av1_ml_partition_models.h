/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_ENCODER_AV1_ML_PARTITION_MODELS_H_
#define AOM_AV1_ENCODER_AV1_ML_PARTITION_MODELS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "av1/encoder/ml.h"

// !!! ALL MODEL DATA BELOW IS NOT CORRECT FOR AV1 AND WILL BE REPLACED
// BEFORE ML_BASED_PARTITION IS ENABLED !!!

// TODO(kyslov): Replace with proper weights after training AV1 models

#define FEATURES 6
static const float av1_var_part_nn_weights_64_layer0[FEATURES * 8] = {
  -0.249572f, 0.205532f,  -2.175608f, 1.094836f,  -2.986370f, 0.193160f,
  -0.143823f, 0.378511f,  -1.997788f, -2.166866f, -1.930158f, -1.202127f,
  -0.611875f, -0.506422f, -0.432487f, 0.071205f,  0.578172f,  -0.154285f,
  -0.051830f, 0.331681f,  -1.457177f, -2.443546f, -2.000302f, -1.389283f,
  0.372084f,  -0.464917f, 2.265235f,  2.385787f,  2.312722f,  2.127868f,
  -0.403963f, -0.177860f, -0.436751f, -0.560539f, 0.254903f,  0.193976f,
  -0.305611f, 0.256632f,  0.309388f,  -0.437439f, 1.702640f,  -5.007069f,
  -0.323450f, 0.294227f,  1.267193f,  1.056601f,  0.387181f,  -0.191215f,
};

static const float av1_var_part_nn_bias_64_layer0[8] = {
  -0.044396f, -0.938166f, 0.000000f,  -0.916375f,
  1.242299f,  0.000000f,  -0.405734f, 0.014206f,
};

static const float av1_var_part_nn_weights_64_layer1[8] = {
  1.635945f,  0.979557f,  0.455315f, 1.197199f,
  -2.251024f, -0.464953f, 1.378676f, -0.111927f,
};

static const float av1_var_part_nn_bias_64_layer1[1] = {
  -0.37972447f,
};

static const NN_CONFIG av1_var_part_nnconfig_64 = {
  FEATURES,  // num_inputs
  1,         // num_outputs
  1,         // num_hidden_layers
  {
      8,
  },  // num_hidden_nodes
  {
      av1_var_part_nn_weights_64_layer0,
      av1_var_part_nn_weights_64_layer1,
  },
  {
      av1_var_part_nn_bias_64_layer0,
      av1_var_part_nn_bias_64_layer1,
  },
};

static const float av1_var_part_nn_weights_32_layer0[FEATURES * 8] = {
  0.067243f,  -0.083598f, -2.191159f, 2.726434f,  -3.324013f, 3.477977f,
  0.323736f,  -0.510199f, 2.960693f,  2.937661f,  2.888476f,  2.938315f,
  -0.307602f, -0.503353f, -0.080725f, -0.473909f, -0.417162f, 0.457089f,
  0.665153f,  -0.273210f, 0.028279f,  0.972220f,  -0.445596f, 1.756611f,
  -0.177892f, -0.091758f, 0.436661f,  -0.521506f, 0.133786f,  0.266743f,
  0.637367f,  -0.160084f, -1.396269f, 1.020841f,  -1.112971f, 0.919496f,
  -0.235883f, 0.651954f,  0.109061f,  -0.429463f, 0.740839f,  -0.962060f,
  0.299519f,  -0.386298f, 1.550231f,  2.464915f,  1.311969f,  2.561612f,
};

static const float av1_var_part_nn_bias_32_layer0[8] = {
  0.368242f, 0.736617f, 0.000000f,  0.757287f,
  0.000000f, 0.613248f, -0.776390f, 0.928497f,
};

static const float av1_var_part_nn_weights_32_layer1[8] = {
  0.939884f, -2.420850f, -0.410489f, -0.186690f,
  0.063287f, -0.522011f, 0.484527f,  -0.639625f,
};

static const float av1_var_part_nn_bias_32_layer1[1] = {
  -0.6455006f,
};

static const NN_CONFIG av1_var_part_nnconfig_32 = {
  FEATURES,  // num_inputs
  1,         // num_outputs
  1,         // num_hidden_layers
  {
      8,
  },  // num_hidden_nodes
  {
      av1_var_part_nn_weights_32_layer0,
      av1_var_part_nn_weights_32_layer1,
  },
  {
      av1_var_part_nn_bias_32_layer0,
      av1_var_part_nn_bias_32_layer1,
  },
};

static const float av1_var_part_nn_weights_16_layer0[FEATURES * 8] = {
  0.742567f,  -0.580624f, -0.244528f, 0.331661f,  -0.113949f, -0.559295f,
  -0.386061f, 0.438653f,  1.467463f,  0.211589f,  0.513972f,  1.067855f,
  -0.876679f, 0.088560f,  -0.687483f, -0.380304f, -0.016412f, 0.146380f,
  0.015318f,  0.000351f,  -2.764887f, 3.269717f,  2.752428f,  -2.236754f,
  0.561539f,  -0.852050f, -0.084667f, 0.202057f,  0.197049f,  0.364922f,
  -0.463801f, 0.431790f,  1.872096f,  -0.091887f, -0.055034f, 2.443492f,
  -0.156958f, -0.189571f, -0.542424f, -0.589804f, -0.354422f, 0.401605f,
  0.642021f,  -0.875117f, 2.040794f,  1.921070f,  1.792413f,  1.839727f,
};

static const float av1_var_part_nn_bias_16_layer0[8] = {
  2.901234f, -1.940932f, -0.198970f, -0.406524f,
  0.059422f, -1.879207f, -0.232340f, 2.979821f,
};

static const float av1_var_part_nn_weights_16_layer1[8] = {
  -0.528731f, 0.375234f, -0.088422f, 0.668629f,
  0.870449f,  0.578735f, 0.546103f,  -1.957207f,
};

static const float av1_var_part_nn_bias_16_layer1[1] = {
  -1.95769405f,
};

static const NN_CONFIG av1_var_part_nnconfig_16 = {
  FEATURES,  // num_inputs
  1,         // num_outputs
  1,         // num_hidden_layers
  {
      8,
  },  // num_hidden_nodes
  {
      av1_var_part_nn_weights_16_layer0,
      av1_var_part_nn_weights_16_layer1,
  },
  {
      av1_var_part_nn_bias_16_layer0,
      av1_var_part_nn_bias_16_layer1,
  },
};
#undef FEATURES

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_AV1_ML_PARTITION_MODELS_H_
