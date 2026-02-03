#----------------------------------------------------------------------------
#  SPDX-FileCopyrightText: Copyright 2021-2022, 2024 Arm Limited and/or its
#  affiliates <open-source-office@arm.com>
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# [Common Configuration]
# ---------------------------------------------------------------------------

# Append the API to use for this use case
list(APPEND ${use_case}_API_LIST "kws" "img_class")

USER_OPTION(${use_case}_MODEL_IN_EXT_FLASH "Run model from external flash"
    ON
    BOOL)

# Activation Buffer Size: set based on the larger of the two models
USER_OPTION(${use_case}_ACTIVATION_BUF_SZ "Activation buffer size for the chosen model"
    0x00172000 
    STRING)

USER_OPTION(${use_case}_USE_APP_MENU "Show application menu"
    OFF
    BOOL)

# User input file configuration (BMP input for Image Classification)
set_input_file_path_user_option(".bmp" ${use_case})

set(SE_SERVICES_SUPPORT ON CACHE BOOL "Enables SE Services initialization.")
message(STATUS "alif_kws_img: SE_SERVICES_SUPPORT: ${SE_SERVICES_SUPPORT}")

set(${use_case}_COMPILE_DEFS
    USE_APP_MENU=$<BOOL:${${use_case}_USE_APP_MENU}>
    $<$<BOOL:${SE_SERVICES_SUPPORT}>:SE_SERVICES_SUPPORT>
)

# ---------------------------------------------------------------------------
# [Image Classification Configuration]
# ---------------------------------------------------------------------------

USER_OPTION(${use_case}_IMAGE_SIZE "Square image size in pixels. Images will be resized to this size."
    224
    STRING)

# Add _IMG suffix to avoid variable name conflicts
USER_OPTION(${use_case}_LABELS_TXT_FILE_IMG "Labels' txt file for the Image model."
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/img_class/labels/labels_mobilenet_v2_1.0_224.txt
    FILEPATH)

# Separate score threshold for Image Classification
USER_OPTION(${use_case}_MODEL_SCORE_THRESHOLD_IMG "Score threshold for Image Classification."
    0.5
    STRING)

# ---------------------------------------------------------------------------
# [KWS Configuration]
# ---------------------------------------------------------------------------

# Add _KWS suffix to avoid variable name conflicts
USER_OPTION(${use_case}_LABELS_TXT_FILE_KWS "Labels' txt file for the KWS model."
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/kws/labels/micronet_kws_labels.txt
    FILEPATH)

USER_OPTION(${use_case}_AUDIO_RATE "Specify the target sampling rate. Default is 16000."
    16000
    STRING)

# Separate score threshold for KWS
USER_OPTION(${use_case}_MODEL_SCORE_THRESHOLD_KWS "Score threshold for KWS."
    0.5
    STRING)

# ---------------------------------------------------------------------------
# [Code Generation: Input Images]
# ---------------------------------------------------------------------------

generate_images_code("${${use_case}_FILE_PATH}"
                     ${SAMPLES_GEN_DIR}
                     "${${use_case}_IMAGE_SIZE}")

# ---------------------------------------------------------------------------
# [Code Generation: Labels]
# ---------------------------------------------------------------------------

# 1. KWS Labels
generate_labels_code(
    INPUT           "${${use_case}_LABELS_TXT_FILE_KWS}"
    DESTINATION_SRC ${SRC_GEN_DIR}
    DESTINATION_HDR ${INC_GEN_DIR}
    OUTPUT_FILENAME "Labels_micronetkws"
    NAMESPACE       "arm" "app" "kws"
)

# 2. Image Labels
generate_labels_code(
    INPUT           "${${use_case}_LABELS_TXT_FILE_IMG}"
    DESTINATION_SRC ${SRC_GEN_DIR}
    DESTINATION_HDR ${INC_GEN_DIR}
    OUTPUT_FILENAME "Labels_mobilenet"
    NAMESPACE       "arm" "app" "img_class"
)

# ---------------------------------------------------------------------------
# [Code Generation: Models]
# ---------------------------------------------------------------------------

if (ETHOS_U_NPU_ENABLED)
    set(DEFAULT_MODEL_PATH_KWS      ${RESOURCES_PATH}/kws/kws_micronet_m_vela_${ETHOS_U_NPU_CONFIG_ID}.tflite)
    set(DEFAULT_MODEL_PATH_IMG      ${DEFAULT_MODEL_DIR}/mobilenet_v2_1.0_224_INT8_vela_${ETHOS_U_NPU_CONFIG_ID}.tflite)
else()
    set(DEFAULT_MODEL_PATH_KWS      ${RESOURCES_PATH}/kws/kws_micronet_m.tflite)
    set(DEFAULT_MODEL_PATH_IMG      ${DEFAULT_MODEL_DIR}/mobilenet_v2_1.0_224_INT8.tflite)
endif()


USER_OPTION(${use_case}_MODEL_TFLITE_PATH_KWS "NN models file for KWS."
    ${DEFAULT_MODEL_PATH_KWS}
    FILEPATH)

USER_OPTION(${use_case}_MODEL_TFLITE_PATH_IMG "NN models file for Image Classification."
    ${DEFAULT_MODEL_PATH_IMG}
    FILEPATH)


set(EXTRA_MODEL_CODE_KWS
    "/* Model parameters for ${use_case} - KWS */"
    "extern const int   g_FrameLength    = 640"
    "extern const int   g_FrameStride    = 320"
    "extern const int   g_AudioRate      = ${${use_case}_AUDIO_RATE}"
    "extern const float g_ScoreThreshold = ${${use_case}_MODEL_SCORE_THRESHOLD_KWS}"
)

set(EXTRA_MODEL_CODE_IMG
    "/* Model parameters for ${use_case} - IMG */"
    "extern const float g_ScoreThreshold = ${${use_case}_MODEL_SCORE_THRESHOLD_IMG}"
)

# 1. Generate KWS Model Code
generate_tflite_code(
    MODEL_PATH  ${${use_case}_MODEL_TFLITE_PATH_KWS}
    DESTINATION ${SRC_GEN_DIR}
    EXPRESSIONS ${EXTRA_MODEL_CODE_KWS}
    NAMESPACE   "arm" "app" "kws"
)

# 2. Generate Image Model Code
generate_tflite_code(
    MODEL_PATH  ${${use_case}_MODEL_TFLITE_PATH_IMG}
    DESTINATION ${SRC_GEN_DIR}
    EXPRESSIONS ${EXTRA_MODEL_CODE_IMG}
    NAMESPACE   "arm" "app" "img_class"
)
