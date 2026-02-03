/* This file was ported to work on Alif Semiconductor devices. */

/* Copyright (C) 2023 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/*
 * Copyright (c) 2021-2022 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "UseCaseHandler.hpp"
#include "KwsClassifier.hpp"
#include "MicroNetKwsModel.hpp"
#include "KwsResult.hpp"
#include "KwsProcessing.hpp"
#include "Classifier.hpp"
#include "MobileNetModel.hpp"
#include "ImgClassProcessing.hpp"
#include "AudioUtils.hpp"
#include "ImageUtils.hpp"
#include "UseCaseCommonUtils.hpp"
#include "hal.h"
#include "timer_alif.h"
#include "log_macros.h"
#include "sys_utils.h"
#include <vector>

#ifdef SE_SERVICES_SUPPORT
#include "services_lib_api.h"
#include "services_main.h"
#if defined(M55_HE) || defined(RTSS_HE)
extern uint32_t hp_comms_handle;
#else
extern uint32_t he_comms_handle;
#endif
m55_data_payload_t mhu_data;
#endif

using arm::app::ApplicationContext;
using arm::app::Model;
using arm::app::Profiler;
using arm::app::ClassificationResult;
using arm::app::KwsClassifier;
using arm::app::KwsPreProcess;
using arm::app::KwsPostProcess;
using arm::app::MicroNetKwsModel;
using ImgClassClassifier = arm::app::Classifier;
using arm::app::ImgClassPreProcess;
using arm::app::ImgClassPostProcess;
using arm::app::MobileNetModel;

#define AUDIO_SAMPLES 16000 
#define AUDIO_STRIDE 8000   
#define RESULTS_MEMORY 8

static int16_t audio_inf[AUDIO_SAMPLES + AUDIO_STRIDE];

namespace alif {
namespace app {

    /* Forward Declarations */
    static bool PresentKwsResults(const std::vector<arm::app::kws::KwsResult>& results);
    static bool PresentImgResults(const std::vector<ClassificationResult>& results);

#ifdef SE_SERVICES_SUPPORT
    static std::string last_label;
    static void send_msg_if_needed(arm::app::kws::KwsResult &result) {
        mhu_data.id = 2; 
        if (result.m_resultVec.empty()) {
            last_label.clear();
            return;
        }
        ClassificationResult classification = result.m_resultVec[0];
        if (classification.m_label != last_label) {
            if (classification.m_label == "_silence_") {
                strcpy(mhu_data.msg, classification.m_label.c_str());
                __DMB();
#if defined(M55_HE) || defined(RTSS_HE)
                SERVICES_send_msg(hp_comms_handle, LocalToGlobal(&mhu_data));
#else
                SERVICES_send_msg(he_comms_handle, LocalToGlobal(&mhu_data));
#endif
            }
            last_label = classification.m_label;
        }
    }
#endif

    /**
     * @brief   Image Handler
     */
    bool ClassifyImageHandler(ApplicationContext& ctx)
    {
        auto& profiler = ctx.Get<Profiler&>("profiler");
        auto& model    = ctx.Get<Model&>("imgModel");
        uint8_t* tensorArena = ctx.Get<uint8_t*>("tensorArena");
        size_t tensorArenaSize = ctx.Get<size_t>("tensorArenaSize");
        uint8_t* imgModelPtr = ctx.Get<uint8_t*>("imgModelPtr");
        size_t imgModelLen = ctx.Get<size_t>("imgModelLen");

        /* Init Image Model */
        if (!model.Init(tensorArena, tensorArenaSize, imgModelPtr, imgModelLen)) {
             printf_err("Image Model Init failed\n");
             return false;
        }

        TfLiteTensor* inputTensor  = model.GetInputTensor(0);
        TfLiteTensor* outputTensor = model.GetOutputTensor(0);
        TfLiteIntArray* inputShape = model.GetInputShape(0);
        const uint32_t nCols       = inputShape->data[MobileNetModel::ms_inputColsIdx];
        const uint32_t nRows       = inputShape->data[MobileNetModel::ms_inputRowsIdx];
        const uint32_t nChannels   = inputShape->data[MobileNetModel::ms_inputChannelsIdx];

        /* Create Process Objects Locally */
        ImgClassPreProcess preProcess = ImgClassPreProcess(inputTensor, model.IsDataSigned());
        std::vector<ClassificationResult> results;
        ImgClassPostProcess postProcess = ImgClassPostProcess(outputTensor,
                                ctx.Get<ImgClassClassifier&>("imgClassifier"),
                                ctx.Get<std::vector<std::string>&>("imgLabels"),
                                results);
        
        hal_camera_stop();
        hal_camera_init();
        if (!hal_camera_configure(nCols, nRows, HAL_CAMERA_MODE_SINGLE_FRAME, HAL_CAMERA_COLOUR_FORMAT_RGB888)) {
            printf_err("Failed to configure camera.\n");
            return false;
        }

        int processed_count = 0;
        for (int i = 0; i < 5; i++) { 
            hal_lcd_clear(COLOR_BLACK);
            hal_camera_start();
            uint32_t capturedFrameSize = 0;
            const uint8_t* imgSrc = hal_camera_get_captured_frame(&capturedFrameSize);
            // If no image is available from the camera, break the loop.
            if (!imgSrc || !capturedFrameSize) {
                info("No more images available (Index end).\n");
                break;
            }

            processed_count++;
            
            hal_lcd_display_image(imgSrc, nCols, nRows, nChannels, 10, 35, 2);
            
            const size_t imgSz = inputTensor->bytes < capturedFrameSize ? inputTensor->bytes : capturedFrameSize;

            /* Run Inference */
            uint32_t start = Get_SysTick_Cycle_Count32();
            if (!preProcess.DoPreProcess(imgSrc, imgSz)) return false;
            info("Preprocessing time = %.3f ms\n", (double)(Get_SysTick_Cycle_Count32() - start) / SystemCoreClock * 1000);

            start = Get_SysTick_Cycle_Count32();
            if (!RunInference(model, profiler)) return false;
            info("Inference time = %.3f ms\n", (double)(Get_SysTick_Cycle_Count32() - start) / SystemCoreClock * 1000);

            start = Get_SysTick_Cycle_Count32();
            if (!postProcess.DoPostProcess()) return false;
            info("Postprocessing time = %.3f ms\n", (double)(Get_SysTick_Cycle_Count32() - start) / SystemCoreClock * 1000);
            PresentImgResults(results);
        }

        hal_camera_stop();

        // If no images were processed (e.g., the image source was empty), a small
        // delay is required here. Returning to the KWS handler too quickly and
        // re-initializing the audio driver can lead to hardware instability.
        if (processed_count == 0) {
            info("Images skipped! Waiting for hardware stability...\n");
            // Busy-wait for an arbitrary amount of time.
            for(volatile int k=0; k<50000000; k++); 
        }
        
        return true;
    }

    /**
     * @brief   KWS Handler (Main)
     */
    bool ClassifyAudioHandler(ApplicationContext& ctx, bool oneshot)
    {
        auto& profiler = ctx.Get<Profiler&>("profiler");
        auto& model = ctx.Get<Model&>("kwsModel");
        
        const auto mfccFrameLength = ctx.Get<int>("frameLength");
        const auto mfccFrameStride = ctx.Get<int>("frameStride");
        const auto audioRate = ctx.Get<int>("audioRate");
        const auto scoreThreshold = ctx.Get<float>("kwsScoreThreshold");

        uint8_t* tensorArena = ctx.Get<uint8_t*>("tensorArena");
        size_t tensorArenaSize = ctx.Get<size_t>("tensorArenaSize");
        uint8_t* kwsModelPtr = ctx.Get<uint8_t*>("kwsModelPtr");
        size_t kwsModelLen = ctx.Get<size_t>("kwsModelLen");

        /* 1. Initial KWS model initialization */
        if (!model.Init(tensorArena, tensorArenaSize, kwsModelPtr, kwsModelLen)) {
             printf_err("KWS Model Init failed\n");
             return false;
        }

        int index = 0;
        std::vector<arm::app::kws::KwsResult> infResults;
        
        static bool audio_inited = false;
        if (!audio_inited) {
            if (hal_audio_alif_init(audioRate) != 0) return false;
            audio_inited = true;
        }

        hal_get_audio_data(audio_inf + AUDIO_SAMPLES, AUDIO_STRIDE);

        /* Main audio processing loop */
        do {
            /* Re-acquire tensor pointers at the start of each loop. This is necessary
             * because the model is re-initialized when switching back from the image
             * handler, which can invalidate previous tensor pointers. */
            TfLiteTensor* inputTensor = model.GetInputTensor(0);
            TfLiteTensor* outputTensor = model.GetOutputTensor(0);
            TfLiteIntArray* inputShape = model.GetInputShape(0);
            
            const uint32_t numMfccFeatures = inputShape->data[MicroNetKwsModel::ms_inputColsIdx];
            const uint32_t numMfccFrames = inputShape->data[MicroNetKwsModel::ms_inputRowsIdx];
            const float secondsPerSample = 1.0f / audioRate;

            /* Stack-allocate pre/post-processing objects in each iteration. This
             * avoids dangling references to tensors, as the underlying model
             * object is re-initialized during the loop. */
            KwsPreProcess preProcess(inputTensor, numMfccFeatures, numMfccFrames, mfccFrameLength, mfccFrameStride);
            std::vector<ClassificationResult> singleInfResult;
            KwsPostProcess postProcess(outputTensor, ctx.Get<KwsClassifier &>("kwsClassifier"),
                                       ctx.Get<std::vector<std::string>&>("kwsLabels"), singleInfResult);

            int err = hal_wait_for_audio();
            if (err) {
                printf_err("hal_wait_for_audio failed with error: %d\n", err);
                return false;
            }

            std::copy(audio_inf + AUDIO_STRIDE, audio_inf + AUDIO_STRIDE + AUDIO_SAMPLES, audio_inf);
            hal_get_audio_data(audio_inf + AUDIO_SAMPLES, AUDIO_STRIDE);
            hal_audio_alif_preprocessing(audio_inf + AUDIO_SAMPLES - AUDIO_STRIDE, AUDIO_STRIDE);

            /* Run Inference and collect results */
            uint32_t start = Get_SysTick_Cycle_Count32();
            if (!preProcess.DoPreProcess(audio_inf, index)) {
                printf_err("Pre-processing failed\n");
                return false;
            }
            info("Preprocessing time = %.3f ms\n", (double)(Get_SysTick_Cycle_Count32() - start) / SystemCoreClock * 1000);

            start = Get_SysTick_Cycle_Count32();
            if (!RunInference(model, profiler)) {
                printf_err("Inference failed\n");
                return false;
            }
            info("Inference time = %.3f ms\n", (double)(Get_SysTick_Cycle_Count32() - start) / SystemCoreClock * 1000);

            start = Get_SysTick_Cycle_Count32();
            if (!postProcess.DoPostProcess()) {
                printf_err("Post-processing failed\n");
                return false;
            }
            info("Postprocessing time = %.3f ms\n", (double)(Get_SysTick_Cycle_Count32() - start) / SystemCoreClock * 1000);


            if (infResults.size() == RESULTS_MEMORY) infResults.erase(infResults.begin());
            infResults.emplace_back(arm::app::kws::KwsResult(singleInfResult,
                    index * secondsPerSample * preProcess.m_audioDataStride, index, scoreThreshold));

#ifdef SE_SERVICES_SUPPORT
            send_msg_if_needed(infResults.back());
#endif
            hal_lcd_clear(COLOR_BLACK);
            PresentKwsResults(infResults);

            // Log inference results to the serial port.
            if (!singleInfResult.empty()) {
                info("Inference #%d: Label=%s, Score=%.2f%%\n", 
                     index, singleInfResult[0].m_label.c_str(), singleInfResult[0].m_normalisedVal * 100);
            } else {
                info("Inference #%d: None\n", index);
            }

            /* Trigger Logic: On detecting the keyword, switch to the image handler. */
            if (!singleInfResult.empty()) {
                if (singleInfResult[0].m_normalisedVal > scoreThreshold && 
                    singleInfResult[0].m_label == "_silence_") {
                    
                    info("Trigger '_silence_' detected! Switching to Image...\n");

                    hal_audio_stop();
                    
                    /* 1. Run the Image Handler. This will re-initialize the tensor
                     *    arena for the image classification model. */
                    if (!ClassifyImageHandler(ctx)) {
                        printf_err("Image Handler failed\n");
                    }

                    /* 2. Re-initialize the KWS model after the image handler returns.
                     *    This is critical to reclaim the tensor arena for KWS. */
                    info("Returning to KWS... Re-initializing model.\n");
                    if (!model.Init(tensorArena, tensorArenaSize, kwsModelPtr, kwsModelLen)) {
                        printf_err("KWS Re-Init failed\n");
                        return false;
                    }

                    info("Restarting audio capture...\n");
                    hal_get_audio_data(audio_inf + AUDIO_SAMPLES, AUDIO_STRIDE);
                    
                    /* 3. Continue the loop. The pre/post-processing objects will be
                     *    re-created at the top of the loop with fresh tensor pointers. */
                    hal_lcd_clear(COLOR_BLACK);
                    continue; 
                }
            }
            index++;
        } while (!oneshot);

        return true;
    }

    /* Helper Implementation */
    static bool PresentKwsResults(const std::vector<arm::app::kws::KwsResult>& results)
    {
        constexpr uint32_t startX = 20, startY = 30, yIncr = 16;
        hal_lcd_set_text_color(COLOR_GREEN);
        uint32_t row = startY + 2 * yIncr;
        for (const auto& result : results) {
            std::string label = "<none>";
            float score = 0.f;
            if (!result.m_resultVec.empty()) {
                label = result.m_resultVec[0].m_label;
                score = result.m_resultVec[0].m_normalisedVal;
            }
            std::string str = "@" + std::to_string(result.m_timeStamp) + "s: " + 
                              label + " (" + std::to_string((int)(score * 100)) + "%)";
            hal_lcd_display_text(str.c_str(), str.size(), startX, row, false);
            row += yIncr;
        }
        return true;
    }

    static bool PresentImgResults(const std::vector<ClassificationResult>& results)
    {
        constexpr uint32_t startX = 150, startY = 60;
        hal_lcd_set_text_color(COLOR_GREEN);
        std::string str = "Label: ";
        if (results.empty()) str += "None";
        else str += results[0].m_label + " (" + std::to_string((int)(results[0].m_normalisedVal * 100)) + "%)";
        hal_lcd_display_text(str.c_str(), str.size(), startX, startY, false);
        return true;
    }

} /* namespace app */
} /* namespace alif */