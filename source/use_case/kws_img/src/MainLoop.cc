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

    arm::app::ApplicationContext caseContext;
    arm::app::Profiler profiler{"kws_img"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);

    // -------------------------------------------------------------------------
    // KWS Resource Configuration
    // -------------------------------------------------------------------------
    arm::app::MicroNetKwsModel kwsModel;
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
    // Image Classification Configurations
    // -------------------------------------------------------------------------
    arm::app::MobileNetModel imgModel;
    caseContext.Set<arm::app::Model&>("imgModel", imgModel);

    caseContext.Set<float>("imgScoreThreshold", arm::app::img_class::g_ScoreThreshold);

    arm::app::Classifier imgClassifier;
    caseContext.Set<arm::app::Classifier&>("imgClassifier", imgClassifier);

    std::vector<std::string> imgLabels;
    arm::app::img_class::GetLabelsVector(imgLabels);
    caseContext.Set<const std::vector<std::string>&>("imgLabels", imgLabels);

    // -------------------------------------------------------------------------
    // Shared Resource Registration
    // -------------------------------------------------------------------------
    caseContext.Set<uint8_t*>("tensorArena", arm::app::tensorArena);
    caseContext.Set<size_t>("tensorArenaSize", sizeof(arm::app::tensorArena));
    
    caseContext.Set<uint8_t*>("kwsModelPtr", arm::app::kws::GetModelPointer());
    caseContext.Set<size_t>("kwsModelLen", arm::app::kws::GetModelLen());

    caseContext.Set<uint8_t*>("imgModelPtr", arm::app::img_class::GetModelPointer());
    caseContext.Set<size_t>("imgModelLen", arm::app::img_class::GetModelLen());

    info("Starting KWS -> Image Classification Loop...\n");
    
    // oneshot = false (infinite loop mode)
    alif::app::ClassifyAudioHandler(caseContext, false);

    info("Main loop terminated.\n");
}