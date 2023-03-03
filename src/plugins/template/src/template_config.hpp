// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace TemplatePlugin {

// ! [configuration:header]
using ConfigMap = std::map<std::string, ov::Any>;

struct Configuration {
    Configuration();
    Configuration(const Configuration&) = default;
    Configuration(Configuration&&) = default;
    Configuration& operator=(const Configuration&) = default;
    Configuration& operator=(Configuration&&) = default;

    explicit Configuration(const ConfigMap& config,
                           const Configuration& defaultCfg = {},
                           const bool throwOnUnsupported = true);

    ov::Any Get(const std::string& name) const;

    // Plugin configuration parameters

    int deviceId = 0;
    bool perfCount = true;
    ov::threading::IStreamsExecutor::Config _streamsExecutorConfig;
    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::UNDEFINED;
};
// ! [configuration:header]

}  //  namespace TemplatePlugin
