// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(paged_causal_conv1d_ref)
(__global INPUT0_TYPE* input_embeds,
 __global INPUT1_TYPE* conv_state_table,
 __global INPUT2_TYPE* conv_weight,
 __global INPUT3_TYPE* conv_bias,
 __global INPUT4_TYPE* subsequence_begins,
 __global INPUT5_TYPE* block_indices,
 __global INPUT6_TYPE* block_indices_begins,
 __global INPUT7_TYPE* past_lens,
 __global INPUT8_TYPE* cache_interval,
#if HAS_TREE_MASK
 __global INPUT9_TYPE* qq_bias,
 __global INPUT10_TYPE* qq_bias_begins,
#endif
 __global OUTPUT_TYPE* output_embeds,
 int seq_count,
 int hidden_size,
 int input_token_stride,
 int input_hidden_stride,
 int state_block_stride,
 int state_hidden_stride,
 int state_kernel_stride,
 int weight_hidden_stride,
 int weight_kernel_stride,
 int bias_hidden_stride,
 int output_token_stride,
 int output_hidden_stride,
 int num_blocks) {
    const int seq = (int)get_global_id(0);
    const int h = (int)get_global_id(1);

    if (seq >= seq_count || h >= hidden_size)
        return;

    const int token_begin = subsequence_begins[seq];
    const int token_end = subsequence_begins[seq + 1];
    const int blk_begin = block_indices_begins[seq];
    const int blk_end = block_indices_begins[seq + 1];

    if (token_begin < 0 || token_end <= token_begin)
        return;

    const int seq_tokens = token_end - token_begin;
    const int block_span = blk_end - blk_begin;
#if HAS_TREE_MASK
    const int qq_begin = qq_bias_begins[seq];
    const int qq_end = qq_bias_begins[seq + 1];
    const int tree_mode = qq_end > qq_begin;
    if (tree_mode && qq_end - qq_begin != seq_tokens * seq_tokens)
        return;
#endif

    if (blk_end <= blk_begin ||
#if HAS_TREE_MASK
        (tree_mode && block_span < seq_tokens + 1) ||
#endif
        block_span <= 1) {
        for (int t = 0; t < seq_tokens; t++) {
            const int out_off = (token_begin + t) * output_token_stride + h * output_hidden_stride;
            output_embeds[out_off] = TO_OUTPUT_TYPE(0.0f);
        }
        return;
    }

    const int read_physical_block = block_indices[blk_begin];
    if (read_physical_block < 0 || read_physical_block >= num_blocks) {
        for (int t = 0; t < seq_tokens; t++) {
            const int out_off = (token_begin + t) * output_token_stride + h * output_hidden_stride;
            output_embeds[out_off] = TO_OUTPUT_TYPE(0.0f);
        }
        return;
    }

    const int seq_interval = cache_interval[seq];
    const int prev_nums = (seq_interval > 0) ? (past_lens[seq] % seq_interval) : 0;

    float state[KERNEL_SIZE];

    const int read_state_base = read_physical_block * state_block_stride + h * state_hidden_stride;
    for (int k = 0; k < KERNEL_SIZE; k++) {
        state[k] = convert_float(conv_state_table[read_state_base + k * state_kernel_stride]);
    }

    float bias_val = 0.0f;
#if HAS_BIAS
    bias_val = convert_float(conv_bias[h * bias_hidden_stride]);
#endif

    for (int t = 0; t < seq_tokens; t++) {
        const int token_idx = token_begin + t;

#if HAS_TREE_MASK
        if (tree_mode) {
            if (qq_bias[qq_begin + t * seq_tokens + t] == 0)
            return;
            int parent = -1;
            for (int candidate = t - 1; candidate >= 0; candidate--) {
                if (qq_bias[qq_begin + t * seq_tokens + candidate] != 0) {
                    parent = candidate;
                    break;
                }
            }
            const int parent_block = block_indices[blk_begin + (parent < 0 ? 0 : parent + 1)];
            if (parent_block < 0 || parent_block >= num_blocks)
                return;
            const int parent_state_base = parent_block * state_block_stride + h * state_hidden_stride;
            for (int k = 0; k < KERNEL_SIZE; k++) {
                state[k] = convert_float(conv_state_table[parent_state_base + k * state_kernel_stride]);
            }
        }
#endif

        for (int k = 0; k + 1 < KERNEL_SIZE; k++) {
            state[k] = state[k + 1];
        }

        const int in_off = token_idx * input_token_stride + h * input_hidden_stride;
        state[KERNEL_SIZE - 1] = convert_float(input_embeds[in_off]);

        float sum = bias_val;
        const int w_base = h * weight_hidden_stride;
        for (int k = 0; k < KERNEL_SIZE; k++) {
            sum = fma(state[k], convert_float(conv_weight[w_base + k * weight_kernel_stride]), sum);
        }

        const int out_off = token_idx * output_token_stride + h * output_hidden_stride;
        output_embeds[out_off] = TO_OUTPUT_TYPE(sum);

        int slot = 0;
#if HAS_TREE_MASK
        if (tree_mode) {
            slot = t + 1;
        } else
#endif
        {
            const int cached_tokens = prev_nums + (t + 1);
            const int interval_hit = (seq_interval > 0) && ((cached_tokens % seq_interval) == 0);
            const int is_last_token = (t == seq_tokens - 1);
            if (!(interval_hit || is_last_token))
                continue;
            slot = (seq_interval > 0) ? (1 + (cached_tokens - 1) / seq_interval) : 1;
        }
        if (slot >= 1 && slot < block_span) {
            const int physical_block = block_indices[blk_begin + slot];
            if (physical_block >= 0 && physical_block < num_blocks) {
                const int state_base = physical_block * state_block_stride + h * state_hidden_stride;
                for (int k = 0; k < KERNEL_SIZE; k++) {
                    conv_state_table[state_base + k * state_kernel_stride] = TO_INPUT1_TYPE(state[k]);
                }
            }
        }
    }
}
