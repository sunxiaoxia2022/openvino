// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

struct PagedGatedDeltaNetLayerParams {
    int32_t qk_heads;
    int32_t v_heads;
    int32_t qk_head_size;
    int32_t v_head_size;
    std::vector<int32_t> seq_lengths;
    std::vector<int32_t> cache_intervals;
    ov::element::Type element_type;
    std::string target_device;
    bool tree_mode = false;
};

class PagedGatedDeltaNetLayerTest : public testing::WithParamInterface<PagedGatedDeltaNetLayerParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedGatedDeltaNetLayerParams>& obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    std::vector<ov::Tensor> calculate_refs() override;
    std::vector<ov::Tensor> get_plugin_outputs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void SetUp() override;

private:
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    ov::element::Type data_type;
};

}  // namespace ov::test