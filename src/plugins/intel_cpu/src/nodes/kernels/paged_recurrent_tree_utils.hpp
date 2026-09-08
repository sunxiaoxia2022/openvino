// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "openvino/core/except.hpp"

namespace ov::Extensions::Cpu::XARCH::recurrent_tree {

struct TreeMaskInfo {
    const uint8_t* mask = nullptr;
    int32_t node_count = 0;

    explicit operator bool() const {
        return mask != nullptr;
    }
};

inline TreeMaskInfo get_tree_mask(const uint8_t* qq_bias,
                                  const int32_t* qq_bias_begins,
                                  const size_t qq_bias_size,
                                  const size_t qq_bias_begins_size,
                                  const size_t seq,
                                  const int32_t seq_tokens,
                                  const std::string_view op_name) {
    if (qq_bias == nullptr || qq_bias_begins == nullptr || qq_bias_begins_size == 0)
        return {};

    OPENVINO_ASSERT(seq + 1 < qq_bias_begins_size,
                    op_name,
                    " qq_bias_begins does not contain sequence ",
                    seq);
    const int32_t begin = qq_bias_begins[seq];
    const int32_t end = qq_bias_begins[seq + 1];
    OPENVINO_ASSERT(begin >= 0 && end >= begin && static_cast<size_t>(end) <= qq_bias_size,
                    op_name,
                    " has invalid qq_bias range [",
                    begin,
                    ", ",
                    end,
                    ") for sequence ",
                    seq);
    if (begin == end)
        return {};

    const int64_t expected_size = static_cast<int64_t>(seq_tokens) * seq_tokens;
    OPENVINO_ASSERT(static_cast<int64_t>(end - begin) == expected_size,
                    op_name,
                    " expects a square qq_bias matching the sequence token count. Got ",
                    end - begin,
                    " elements for ",
                    seq_tokens,
                    " tokens at sequence ",
                    seq);
    return {qq_bias + begin, seq_tokens};
}

inline int32_t get_parent(const TreeMaskInfo& tree, const int32_t node, const std::string_view op_name) {
    OPENVINO_ASSERT(tree.mask != nullptr && node >= 0 && node < tree.node_count,
                    op_name,
                    " has invalid tree node index ",
                    node);
    OPENVINO_ASSERT(tree.mask[static_cast<size_t>(node) * tree.node_count + node] != 0,
                    op_name,
                    " qq_bias diagonal must be non-zero at node ",
                    node);

    // QQ-bias stores the transitive ancestor mask. Nodes are in topological order, so the
    // closest preceding visible node is the direct parent.
    for (int32_t candidate = node - 1; candidate >= 0; --candidate) {
        if (tree.mask[static_cast<size_t>(node) * tree.node_count + candidate] != 0)
            return candidate;
    }
    return -1;
}

}  // namespace ov::Extensions::Cpu::XARCH::recurrent_tree
