// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/dnnl_thread.hpp>
#include <iostream>
#include <oneapi/dnnl/dnnl_threadpool.hpp>
#include <oneapi/dnnl/dnnl_threadpool_iface.hpp>

#include "cpu_parallel.hpp"
#include "openvino/core/parallel.hpp"
#if OV_THREAD == OV_THREAD_TBB
#    include "tbb/parallel_for.h"
#    include "tbb/task_arena.h"
#endif

namespace ov::intel_cpu {

class ThreadPool : public dnnl::threadpool_interop::threadpool_iface {
private:
    // std::shared_ptr<CpuParallel> m_cpu_parallel = nullptr;
    ov::hint::TbbPartitioner m_partitoner = ov::hint::TbbPartitioner::STATIC;
    size_t m_multiplier = 32;

public:
    ThreadPool() = default;
    ThreadPool(ov::hint::TbbPartitioner partitoner) : m_partitoner(partitoner) {}
    int get_num_threads() const override {
        int num = m_partitoner == ov::hint::TbbPartitioner::STATIC ? parallel_get_max_threads()
                                                                   : parallel_get_max_threads() * m_multiplier;
        return num;
    }
    bool get_in_parallel() const override {
        return 0;
    }
    uint64_t get_flags() const override {
        return 0;
    }
    void parallel_for(int n, const std::function<void(int, int)>& fn) override {
#if OV_THREAD == OV_THREAD_TBB
        if (m_partitoner == ov::hint::TbbPartitioner::STATIC) {
            tbb::parallel_for(
                0,
                n,
                [&](int ithr) {
                    fn(ithr, n);
                },
                tbb::static_partitioner());
        } else {
            tbb::parallel_for(0, n, [&](int ithr) {
                fn(ithr, n);
            });
        }
#endif
    }
};

dnnl::stream make_stream(const dnnl::engine& engine, std::shared_ptr<ThreadPool> thread_pool);

}  // namespace ov::intel_cpu
