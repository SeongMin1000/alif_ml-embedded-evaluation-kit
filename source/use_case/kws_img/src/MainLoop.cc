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
 * SPDX-FileCopyrightText: Copyright 2021-2022, 2024 Arm Limited and/or its
 * affiliates <open-source-office@arm.com>
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
#include "hal.h"                    
#include "UseCaseHandler.hpp"       
#include "UseCaseCommonUtils.hpp"   
#include "log_macros.h"             
#include "BufAttributes.hpp"        

/* KWS Includes */
#include "KwsClassifier.hpp"
#include "MicroNetKwsModel.hpp"

/* Image Classification Includes */
#include "Classifier.hpp"
#include "MobileNetModel.hpp"


namespace arm {
namespace app {
    static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

    namespace kws {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
        extern const int g_AudioRate;
        extern const int g_FrameLength;
        extern const int g_FrameStride;
        extern const float g_ScoreThreshold;

        extern void GetLabelsVector(std::vector<std::string>& labels);
    } 

    namespace img_class {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
        extern const float g_ScoreThreshold;

        extern void GetLabelsVector(std::vector<std::string>& labels);
    }
} /* namespace app */
} /* namespace arm */

void MainLoop()
{
    init_trigger_tx();

    /* Model wrapper objects. */
    arm::app::MicroNetKwsModel kwsModel;
    arm::app::MobileNetModel imgModel;

    /* Load the models. */
    if (!kwsModel.Init(arm::app::tensorArena,
                       sizeof(arm::app::tensorArena),
                       arm::app::kws::GetModelPointer(),
                       arm::app::kws::GetModelLen())) {
        printf_err("Failed to initialise KWS model\n");
        return;
    }

    /* Initialise the image model using the same allocator from KWS
     * to re-use the tensor arena. */
    if (!imgModel.Init(arm::app::tensorArena,
                       sizeof(arm::app::tensorArena),
                       arm::app::img_class::GetModelPointer(),
                       arm::app::img_class::GetModelLen(),
                       kwsModel.GetAllocator())) {
        printf_err("Failed to initialise Image model\n");
        return;
    }

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;
    arm::app::Profiler profiler{"kws_img"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);

    // -------------------------------------------------------------------------
    // KWS Resource Configuration
    // -------------------------------------------------------------------------
    caseContext.Set<arm::app::Model&>("kwsModel", kwsModel);

    caseContext.Set<int>("frameLength", arm::app::kws::g_FrameLength);
    caseContext.Set<int>("frameStride", arm::app::kws::g_FrameStride);
    caseContext.Set<int>("audioRate", arm::app::kws::g_AudioRate);
    caseContext.Set<float>("kwsScoreThreshold", arm::app::kws::g_ScoreThreshold);

    arm::app::KwsClassifier kwsClassifier;
    caseContext.Set<arm::app::KwsClassifier&>("kwsClassifier", kwsClassifier);

    std::vector<std::string> kwsLabels;
    arm::app::kws::GetLabelsVector(kwsLabels);
    caseContext.Set<const std::vector<std::string>&>("kwsLabels", kwsLabels);

    // -------------------------------------------------------------------------
    // Image Classification Resource Configuration
    // -------------------------------------------------------------------------
    caseContext.Set<arm::app::Model&>("imgModel", imgModel);

    caseContext.Set<float>("imgScoreThreshold", arm::app::img_class::g_ScoreThreshold);

    arm::app::Classifier imgClassifier;
    caseContext.Set<arm::app::Classifier&>("imgClassifier", imgClassifier);

    std::vector<std::string> imgLabels;
    arm::app::img_class::GetLabelsVector(imgLabels);
    caseContext.Set<const std::vector<std::string>&>("imgLabels", imgLabels);

    info("Starting KWS -> Image Classification Loop...\n");
    
    // oneshot = false (infinite loop mode)
    alif::app::ClassifyAudioHandler(caseContext, false);

    info("Main loop terminated.\n");
}