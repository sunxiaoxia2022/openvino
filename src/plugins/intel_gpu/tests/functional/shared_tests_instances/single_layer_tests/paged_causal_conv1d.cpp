// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_causal_conv1d.hpp"

namespace ov::test {

TEST_P(PagedCausalConv1DLayerTest, Inference) {
    run();
}

std::vector<PagedCausalConv1DLayerParams> paged_conv1d_test_cases = {
    {8, 3, true, {{4}}, {{1}}, ov::element::f32, "GPU", true},
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedCausalConv1D,
                         PagedCausalConv1DLayerTest,
                         ::testing::ValuesIn(paged_conv1d_test_cases),
                         PagedCausalConv1DLayerTest::getTestCaseName);

}  // namespace ov::test
