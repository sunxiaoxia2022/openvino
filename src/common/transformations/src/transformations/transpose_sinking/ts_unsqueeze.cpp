// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_unsqueeze.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace opset10;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace {

/**
 * @brief Checks that Reshape operation is equal to Unsqueeze:
 * Only 1 dims are inserted, all other dims must be the same.
 * Converts these 1 dims to axes format.
 * @arg reshape Reshape operation.
 * @arg reshape_to_shape 2nd input to Reshape op as a constant.
 * @arg result_axes contains axes which will be unsqueezed.
 */
bool shape_to_unsqueeze_axes(const std::shared_ptr<Node>& reshape,
                             const std::shared_ptr<Constant>& reshape_to_shape,
                             std::vector<size_t>& result_axes) {
    result_axes.clear();
    auto reduction_axes_values = reshape_to_shape->cast_vector<int64_t>();
    // supported the case if Reshape is equal to Unsqueeze
    const auto& new_shape = reduction_axes_values;
    const auto& input_pshape = reshape->get_input_partial_shape(0);
    // todo: support dynamic case
    if (input_pshape.is_dynamic()) {
        return false;
    }

    const auto input_shape = input_pshape.to_shape();
    if (new_shape.size() > input_shape.size()) {
        size_t j = 0;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (j < input_shape.size() && static_cast<int64_t>(input_shape[j]) == new_shape[i]) {
                j++;
            } else if (new_shape[i] != 1) {
                return false;
            } else {
                result_axes.push_back(i);
            }
        }
        if (j != input_shape.size()) {
            // not all input_shape values are in new_shape
            return false;
        }
    } else {
        // another reshape type, not Unsqueeze
        // todo: move this checks in the pattern
        return false;
    }
    return true;
}

/**
 * @brief Converts unsqueeze_axes to actual shape (2nd input) for Reshape operation
 * using the shape of the 1st input to Reshape.
 * @arg input_node 1st input to Reshape op.
 * @arg unsqueeze_axes In case of Reshape op is equal to Unsqueeze, these axes indicate the places where 1 dims have
 * to be inserted.
 */
bool unsqueeze_axes_to_shape(const Output<Node>& input_node,
                             std::vector<size_t> unsqueeze_axes,
                             std::vector<size_t>& to_shape) {
    to_shape.clear();
    const auto& input_pshape = input_node.get_partial_shape();
    if (input_pshape.is_dynamic()) {
        return false;
    }
    const auto& input_shape = input_pshape.get_shape();
    to_shape.resize(input_shape.size() + unsqueeze_axes.size());
    std::sort(unsqueeze_axes.begin(), unsqueeze_axes.end());
    for (size_t i = 0, j = 0, k = 0; i < to_shape.size(); ++i) {
        if (j < unsqueeze_axes.size() && i == unsqueeze_axes[j]) {
            to_shape[i] = 1;
            j++;
        } else if (k < input_shape.size()) {
            to_shape[i] = input_shape[k];
            k++;
        }
    }
    return true;
}
}  // namespace

TSUnsqueezeForward::TSUnsqueezeForward() {
    MATCHER_SCOPE(TSUnsqueezeForward);

    auto transpose_label = wrap_type<Transpose>({any_input(), wrap_type<Constant>()});
    auto unsqueeze_label = wrap_type<Unsqueeze, Reshape>({transpose_label, wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        auto unsqueeze = pattern_to_output.at(unsqueeze_label);

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        auto unsqueeze_axes = as_type_ptr<Constant>(unsqueeze->get_input_node_shared_ptr(1));
        if (!transpose_order || !unsqueeze_axes) {
            return false;
        }

        std::vector<size_t> non_negative_axes;
        if (as_type_ptr<Reshape>(unsqueeze)) {
            auto success = shape_to_unsqueeze_axes(unsqueeze, unsqueeze_axes, non_negative_axes);
            if (!success) {
                return false;
            }
        } else {
            auto rank = unsqueeze->get_output_partial_shape(0).rank();
            non_negative_axes =
                normalize_axes(unsqueeze->get_friendly_name(), unsqueeze_axes->cast_vector<int64_t>(), rank);
        }
        auto ts_order_values = transpose_order->cast_vector<size_t>();

        ts_order_values = GetOrderBeforeReduction(non_negative_axes, ts_order_values);
        auto new_transpose_order =
            Constant::create(transpose_order->get_element_type(), {ts_order_values.size()}, ts_order_values);

        std::shared_ptr<Node> new_unsqueeze;
        if (as_type_ptr<Reshape>(unsqueeze)) {
            std::vector<size_t> new_values;
            auto success = unsqueeze_axes_to_shape(transpose->input_value(0), non_negative_axes, new_values);
            if (!success) {
                return false;
            }
            auto new_const = Constant::create(unsqueeze_axes->get_element_type(), {new_values.size()}, new_values);
            new_unsqueeze = unsqueeze->clone_with_new_inputs({transpose->input_value(0), new_const});
        } else {
            new_unsqueeze = unsqueeze->clone_with_new_inputs({transpose->input_value(0), unsqueeze->input_value(1)});
        }
        auto new_transpose = transpose->clone_with_new_inputs({new_unsqueeze, new_transpose_order});

        replace_node(unsqueeze, new_transpose);
        new_unsqueeze->set_friendly_name(transpose->get_friendly_name());
        new_transpose->set_friendly_name(unsqueeze->get_friendly_name());
        UpdateForwardSinkingAbility(new_transpose);
        register_new_node(new_transpose);
        copy_runtime_info({transpose, unsqueeze}, {new_transpose, new_unsqueeze});

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(unsqueeze_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

TSUnsqueezeBackward::TSUnsqueezeBackward() {
    MATCHER_SCOPE(TSUnsqueezeBackward);

    auto unsqueeze_label =
        wrap_type<Unsqueeze, Reshape>({any_input(), wrap_type<Constant>()}, HasSameOutputTransposeNodes);
    auto transpose_label =
        wrap_type<Transpose>({unsqueeze_label, wrap_type<Constant>()}, [](const Output<Node>& output) -> bool {
            return has_static_rank()(output) && is_sinking_node(output);
        });

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        auto unsqueeze = pattern_to_output.at(unsqueeze_label);

        auto transpose_order = std::dynamic_pointer_cast<Constant>(transpose->get_input_node_shared_ptr(1));
        auto unsqueeze_axes = std::dynamic_pointer_cast<Constant>(unsqueeze->get_input_node_shared_ptr(1));
        if (!transpose_order || !unsqueeze_axes)
            return false;

        std::vector<size_t> non_negative_axes;
        if (as_type_ptr<Reshape>(unsqueeze)) {
            auto success = shape_to_unsqueeze_axes(unsqueeze, unsqueeze_axes, non_negative_axes);
            if (!success) {
                return false;
            }
        } else {
            auto rank = unsqueeze->get_output_partial_shape(0).rank();
            non_negative_axes =
                normalize_axes(unsqueeze->get_friendly_name(), unsqueeze_axes->cast_vector<int64_t>(), rank);
        }

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        auto old_transpose_order_values = transpose_order_values;
        std::vector<size_t> new_values;

        if (non_negative_axes.size() == transpose_order_values.size()) {
            // input is a scalar, we unsqueeze all dims
            // it's enough to eliminate such Transpose
            transpose->output(0).replace(unsqueeze);
            return true;
        }

        for (const auto& axis : non_negative_axes) {
            auto it = std::find(old_transpose_order_values.begin(), old_transpose_order_values.end(), axis);
            if (it != old_transpose_order_values.end()) {
                new_values.push_back(it - old_transpose_order_values.begin());
            }
        }

        transpose_order_values = GetOrderAfterReduction(new_values, transpose_order_values);
        auto new_transpose_order = std::make_shared<Constant>(transpose_order->get_element_type(),
                                                              Shape{transpose_order_values.size()},
                                                              transpose_order_values);

        auto new_transpose = transpose->clone_with_new_inputs({unsqueeze->input_value(0), new_transpose_order});
        if (as_type_ptr<Reshape>(unsqueeze)) {
            std::vector<size_t> to_shape;
            auto success = unsqueeze_axes_to_shape(new_transpose->output(0), new_values, to_shape);
            if (!success) {
                return false;
            }
            new_values = to_shape;
        }
        auto new_const = Constant::create(unsqueeze_axes->get_element_type(), unsqueeze_axes->get_shape(), new_values);
        auto new_unsqueeze = unsqueeze->clone_with_new_inputs({new_transpose, new_const});

        replace_node(transpose, new_unsqueeze);
        copy_runtime_info({transpose, unsqueeze}, {new_transpose, new_unsqueeze});
        new_unsqueeze->set_friendly_name(transpose->get_friendly_name());
        new_transpose->set_friendly_name(unsqueeze->get_friendly_name());
        register_new_node(new_transpose);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
