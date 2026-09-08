// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/paged_recurrent_tree_utils.hpp"

#include <array>

#include "gtest/gtest.h"

namespace ov::intel_cpu::test {

using ov::Extensions::Cpu::XARCH::recurrent_tree::get_parent;
using ov::Extensions::Cpu::XARCH::recurrent_tree::get_tree_mask;

TEST(PagedRecurrentTreeUtilsTest, EmptyMaskDisablesTreeMode) {
    const std::array<int32_t, 2> begins{0, 0};

    const auto tree = get_tree_mask(nullptr, begins.data(), 0, begins.size(), 0, 3, "TestOp");

    EXPECT_FALSE(tree);
}

TEST(PagedRecurrentTreeUtilsTest, FindsDirectParentsFromAncestorMask) {
    // node 0: root; node 1: child of 0; node 2: sibling of 1; node 3: child of 1.
    const std::array<uint8_t, 16> mask{
        1, 0, 0, 0,
        1, 1, 0, 0,
        1, 0, 1, 0,
        1, 1, 0, 1,
    };
    const std::array<int32_t, 2> begins{0, static_cast<int32_t>(mask.size())};

    const auto tree = get_tree_mask(mask.data(), begins.data(), mask.size(), begins.size(), 0, 4, "TestOp");

    ASSERT_TRUE(tree);
    EXPECT_EQ(get_parent(tree, 0, "TestOp"), -1);
    EXPECT_EQ(get_parent(tree, 1, "TestOp"), 0);
    EXPECT_EQ(get_parent(tree, 2, "TestOp"), 0);
    EXPECT_EQ(get_parent(tree, 3, "TestOp"), 1);
}

TEST(PagedRecurrentTreeUtilsTest, RejectsNonSquareSequenceMask) {
    const std::array<uint8_t, 8> mask{};
    const std::array<int32_t, 2> begins{0, static_cast<int32_t>(mask.size())};

    EXPECT_THROW(get_tree_mask(mask.data(), begins.data(), mask.size(), begins.size(), 0, 3, "TestOp"),
                 ov::Exception);
}

TEST(PagedRecurrentTreeUtilsTest, RejectsMissingDiagonal) {
    const std::array<uint8_t, 4> mask{1, 0, 1, 0};
    const std::array<int32_t, 2> begins{0, static_cast<int32_t>(mask.size())};
    const auto tree = get_tree_mask(mask.data(), begins.data(), mask.size(), begins.size(), 0, 2, "TestOp");

    EXPECT_THROW(get_parent(tree, 1, "TestOp"), ov::Exception);
}

}  // namespace ov::intel_cpu::test
