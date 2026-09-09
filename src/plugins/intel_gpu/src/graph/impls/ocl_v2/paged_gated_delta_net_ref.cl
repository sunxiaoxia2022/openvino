// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#ifndef FUSE_QK_L2NORM
#    define FUSE_QK_L2NORM 0
#endif

#ifndef Q_L2_NORM_EPS
#    define Q_L2_NORM_EPS 1e-6f
#endif

#ifndef K_L2_NORM_EPS
#    define K_L2_NORM_EPS 1e-6f
#endif

inline float FUNC(l2norm_scale)(float sum, float extra_scale, float eps) {
    return rsqrt(sum + eps) * extra_scale;
}

#define K_SLICE_SIZE ((K_HEAD_DIM + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE)

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(paged_gated_delta_net_ref)
(__global INPUT0_TYPE* query,
 __global INPUT1_TYPE* key,
 __global INPUT2_TYPE* value,
 __global INPUT3_TYPE* recurrent_state_table,
 __global INPUT4_TYPE* gate,
 __global INPUT5_TYPE* beta,
 __global INPUT6_TYPE* subsequence_begins,
 __global INPUT7_TYPE* block_indices,
 __global INPUT8_TYPE* block_indices_begins,
 __global INPUT9_TYPE* past_lens,
 __global INPUT10_TYPE* cache_interval,
#if HAS_TREE_MASK
 __global INPUT11_TYPE* qq_bias,
 __global INPUT12_TYPE* qq_bias_begins,
#endif
 __global OUTPUT_TYPE* output,
 int num_sequences,
 int query_head_offset,
 int key_head_offset,
 int value_head_offset,
 int q_token_stride,
 int q_head_stride,
 int k_token_stride,
 int k_head_stride,
 int v_token_stride,
 int v_head_stride) {
    const int seq = get_global_id(0);
    const int h = get_global_id(1);
    const int v_block = get_group_id(2);
    const int lid = get_sub_group_local_id();
    const int start_iv = v_block * V_BLOCK_SIZE;

    const int token_begin = subsequence_begins[seq];
    const int token_end = subsequence_begins[seq + 1];
    const int block_begin = block_indices_begins[seq];
    const int past_len = past_lens[seq];
    const int interval = cache_interval[seq];
    const int prev_nums = interval > 0 ? past_len % interval : 0;
#if HAS_TREE_MASK
    const int seq_tokens = token_end - token_begin;
    const int qq_begin = qq_bias_begins[seq];
    const int qq_end = qq_bias_begins[seq + 1];
    const int tree_mode = qq_end > qq_begin;
    const int block_end = block_indices_begins[seq + 1];
    if (tree_mode && (qq_end - qq_begin != seq_tokens * seq_tokens || block_end - block_begin < seq_tokens + 1))
        return;
#endif

    const int group_size = V_HEAD_NUM / K_HEAD_NUM;
    const int hk = h / group_size;
    const int q_head_base = (hk + query_head_offset) * q_head_stride;
    const int k_head_base = (hk + key_head_offset) * k_head_stride;
    const int v_head_base = (h + value_head_offset) * v_head_stride;
    const int state_stride = K_HEAD_DIM * V_HEAD_DIM;

    float state[V_BLOCK_SIZE][K_HEAD_DIM];
    float q_norm[K_SLICE_SIZE];
    float k_norm[K_SLICE_SIZE];

    const int initial_block_id = block_indices[block_begin];
    const int initial_block_base = (initial_block_id * V_HEAD_NUM + h) * state_stride;

#pragma unroll
    for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
        int curr_iv = start_iv + v_idx;
        if (curr_iv >= V_HEAD_DIM)
            continue;

        int base = initial_block_base + curr_iv * K_HEAD_DIM;
        for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
            const int k_idx = lid + ks * SUBGROUP_SIZE;
            if (k_idx < K_HEAD_DIM) {
                state[v_idx][ks] = convert_float(recurrent_state_table[base + k_idx]);
            }
        }
    }

    int token = token_begin;
#if HAS_TREE_MASK
    if (tree_mode) {
        if (lid != 0)
            return;
        for (; token < token_end; token++) {
            const int node = token - token_begin;
            if (qq_bias[qq_begin + node * seq_tokens + node] == 0)
                return;
            int parent = -1;
            for (int candidate = node - 1; candidate >= 0; candidate--) {
                if (qq_bias[qq_begin + node * seq_tokens + candidate] != 0) {
                    parent = candidate;
                    break;
                }
            }
            const int parent_block_id = block_indices[block_begin + (parent < 0 ? 0 : parent + 1)];
            const int parent_block_base = (parent_block_id * V_HEAD_NUM + h) * state_stride;
#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                const int curr_iv = start_iv + v_idx;
                if (curr_iv >= V_HEAD_DIM)
                    continue;
                const int base = parent_block_base + curr_iv * K_HEAD_DIM;
                for (int k_idx = 0; k_idx < K_HEAD_DIM; k_idx++) {
                    state[v_idx][k_idx] = convert_float(recurrent_state_table[base + k_idx]);
                }
            }

            const int q_base = token * q_token_stride + q_head_base;
            const int k_base = token * k_token_stride + k_head_base;
            const int v_base = token * v_token_stride + v_head_base;
            const int g_idx = token * V_HEAD_NUM + h;
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (int k_idx = 0; k_idx < K_HEAD_DIM; k_idx++) {
                const float q_val = convert_float(query[q_base + k_idx]);
                const float k_val = convert_float(key[k_base + k_idx]);
                q_sum = fma(q_val, q_val, q_sum);
                k_sum = fma(k_val, k_val, k_sum);
            }
            float q_scale = SCALE_FACTOR;
            float k_scale = 1.0f;
#if FUSE_QK_L2NORM
            q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, Q_L2_NORM_EPS);
            k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, K_L2_NORM_EPS);
#endif
            const float b_g = exp(convert_float(gate[g_idx]));
            const float b_beta = convert_float(beta[g_idx]);
#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                const int curr_iv = start_iv + v_idx;
                if (curr_iv >= V_HEAD_DIM)
                    continue;
                float h_k = 0.0f;
                for (int k_idx = 0; k_idx < K_HEAD_DIM; k_idx++) {
                    state[v_idx][k_idx] *= b_g;
                    h_k = fma(state[v_idx][k_idx], convert_float(key[k_base + k_idx]) * k_scale, h_k);
                }
                const float update_val = (convert_float(value[v_base + curr_iv]) - h_k) * b_beta;
                float out_val = 0.0f;
                for (int k_idx = 0; k_idx < K_HEAD_DIM; k_idx++) {
                    state[v_idx][k_idx] = fma(convert_float(key[k_base + k_idx]) * k_scale, update_val, state[v_idx][k_idx]);
                    out_val = fma(state[v_idx][k_idx], convert_float(query[q_base + k_idx]) * q_scale, out_val);
                }
                output[(token * V_HEAD_NUM + h) * V_HEAD_DIM + curr_iv] = TO_OUTPUT_TYPE(out_val);
            }
            const int output_block_base = (block_indices[block_begin + node + 1] * V_HEAD_NUM + h) * state_stride;
#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                const int curr_iv = start_iv + v_idx;
                if (curr_iv >= V_HEAD_DIM)
                    continue;
                const int base = output_block_base + curr_iv * K_HEAD_DIM;
                for (int k_idx = 0; k_idx < K_HEAD_DIM; k_idx++) {
                    recurrent_state_table[base + k_idx] = (INPUT3_TYPE)(state[v_idx][k_idx]);
                }
            }
        }
        return;
    }
#endif
    int slot = 1;
    int tokens_to_next_boundary = interval > 0 ? (prev_nums > 0 ? (interval - prev_nums) : interval) : (token_end - token_begin);
    while (token < token_end) {
        const int chunk_end = min(token + tokens_to_next_boundary, token_end);

        int q_base = token * q_token_stride + q_head_base;
        int k_base = token * k_token_stride + k_head_base;
        int v_base = token * v_token_stride + v_head_base;
        int g_idx = token * V_HEAD_NUM + h;

        for (; token < chunk_end; token++, q_base += q_token_stride, k_base += k_token_stride, v_base += v_token_stride, g_idx += V_HEAD_NUM) {
            float q_sum_local = 0.0f;
            float k_sum_local = 0.0f;
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                const int k_idx = lid + ks * SUBGROUP_SIZE;
                if (k_idx < K_HEAD_DIM) {
                    const float q_val = convert_float(query[q_base + k_idx]);
                    const float k_val = convert_float(key[k_base + k_idx]);
                    q_norm[ks] = q_val;
                    k_norm[ks] = k_val;
                    q_sum_local = fma(q_val, q_val, q_sum_local);
                    k_sum_local = fma(k_val, k_val, k_sum_local);
                } else {
                    q_norm[ks] = 0.0f;
                    k_norm[ks] = 0.0f;
                }
            }

            float q_scale = SCALE_FACTOR;
            float k_scale = 1.0f;
#if FUSE_QK_L2NORM
            const float q_sum = sub_group_reduce_add(q_sum_local);
            const float k_sum = sub_group_reduce_add(k_sum_local);

            q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, Q_L2_NORM_EPS);
            k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, K_L2_NORM_EPS);
#endif
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                q_norm[ks] *= q_scale;
                k_norm[ks] *= k_scale;
            }

            const float b_g = exp(convert_float(gate[g_idx]));
            const float b_beta = convert_float(beta[g_idx]);

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                if (curr_iv >= V_HEAD_DIM)
                    continue;
                float h_k_local = 0.0f;
                for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                    state[v_idx][ks] *= b_g;
                    h_k_local = fma(state[v_idx][ks], k_norm[ks], h_k_local);
                }

                const float h_k = sub_group_reduce_add(h_k_local);
                const int v_idx_offset = v_base + curr_iv;
                const float b_v = convert_float(value[v_idx_offset]);
                const float update_val = (b_v - h_k) * b_beta;

                float out_val_local = 0.0f;
                for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                    state[v_idx][ks] = fma(k_norm[ks], update_val, state[v_idx][ks]);
                    out_val_local = fma(state[v_idx][ks], q_norm[ks], out_val_local);
                }
                const float out_val = sub_group_reduce_add(out_val_local);

                if (lid == 0) {
                    const int out_offset = (token * V_HEAD_NUM + h) * V_HEAD_DIM + curr_iv;
                    output[out_offset] = TO_OUTPUT_TYPE(out_val);
                }
            }
        }

        const int block_id = block_indices[block_begin + slot];
        slot++;
        const int block_base = (block_id * V_HEAD_NUM + h) * state_stride;
#pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            if (curr_iv >= V_HEAD_DIM)
                continue;

            int base = block_base + curr_iv * K_HEAD_DIM;
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                const int k_idx = lid + ks * SUBGROUP_SIZE;
                if (k_idx < K_HEAD_DIM) {
                    recurrent_state_table[base + k_idx] = (INPUT3_TYPE)(state[v_idx][ks]);
                }
            }
        }

        if (interval > 0)
            tokens_to_next_boundary = interval;
    }
}
