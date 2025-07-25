// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_desc/cpu_memory_desc_utils.h"

#include <cpu_memory.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory_desc.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "memory_desc/empty_memory_desc.h"
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

using namespace dnnl;

namespace ov::intel_cpu {

DnnlMemoryDescPtr MemoryDescUtils::convertToDnnlMemoryDesc(const MemoryDescPtr& desc) {
    if (MemoryDescType::Blocked == desc->getType()) {
        auto* const cpuDesc = desc->as<CpuBlockedMemoryDesc>();
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(cpuDesc->getPrecision(),
                                                                                cpuDesc->getShape(),
                                                                                cpuDesc->getBlockDims(),
                                                                                cpuDesc->getOrder(),
                                                                                cpuDesc->getOffsetPadding(),
                                                                                cpuDesc->getOffsetPaddingToData(),
                                                                                cpuDesc->getStrides()));
    }
    if (MemoryDescType::Empty == desc->getType()) {
        return DnnlExtensionUtils::makeDescriptor(dnnl::memory::desc());
    }
    if (MemoryDescType::Dnnl & desc->getType()) {
        return std::dynamic_pointer_cast<DnnlMemoryDesc>(desc);
    }
    OPENVINO_THROW("Cannot convert MemoryDesc to DnnlMemoryDesc");
}

DnnlBlockedMemoryDesc MemoryDescUtils::convertToDnnlBlockedMemoryDesc(const MemoryDesc& desc) {
    if (MemoryDescType::DnnlBlocked == desc.getType()) {
        return {*desc.as<DnnlBlockedMemoryDesc>()};
    }
    if (MemoryDescType::Blocked == desc.getType()) {
        const auto* const cpuDesc = desc.as<CpuBlockedMemoryDesc>();
        return {cpuDesc->getPrecision(),
                cpuDesc->getShape(),
                cpuDesc->getBlockDims(),
                cpuDesc->getOrder(),
                cpuDesc->getOffsetPadding(),
                cpuDesc->getOffsetPaddingToData(),
                cpuDesc->getStrides()};
    }
    OPENVINO_THROW("Cannot convert MemoryDesc to DnnlBlockedMemoryDesc");
}

BlockedMemoryDescPtr MemoryDescUtils::convertToBlockedMemoryDesc(const MemoryDescPtr& desc) {
    if (desc->getType() & MemoryDescType::Blocked) {
        return std::dynamic_pointer_cast<BlockedMemoryDesc>(desc);
    }
    OPENVINO_THROW("Cannot convert MemoryDesc to BlockedMemoryDesc");
}

CpuBlockedMemoryDescPtr MemoryDescUtils::generateCpuBlockedMemoryDesc(const ov::SoPtr<ov::ITensor>& tensor) {
    const auto& shape = tensor->get_shape().empty() ? ov::Shape{tensor->get_size()} : tensor->get_shape();

    VectorDims blk_order(shape.size());
    std::iota(blk_order.begin(), blk_order.end(), 0);

    auto element_type = tensor->get_element_type();
    const auto& byte_strides = element_type.bitwidth() >= 8 ? tensor->get_strides() : Strides{};

    VectorDims blk_strides;

    if (byte_strides.empty()) {
        blk_strides = ov::row_major_strides(shape);
    } else if (tensor->get_size() == 0) {
        blk_strides.resize(shape.size());
    } else {
        // ROI tensor need figure out correct blk_strides
        blk_strides.resize(byte_strides.size());
        std::transform(byte_strides.begin(),
                       byte_strides.end(),
                       blk_strides.begin(),
                       [&element_type](size_t byte_stride) {
                           OPENVINO_ASSERT(byte_stride % element_type.size() == 0,
                                           "Limitation: Stride in bytes ",
                                           byte_stride,
                                           " must be divisible by size of element ",
                                           element_type.size());
                           return byte_stride / element_type.size();
                       });
    }

    return std::make_shared<CpuBlockedMemoryDesc>(element_type,
                                                  Shape{shape},
                                                  shape,
                                                  blk_order,
                                                  0UL,
                                                  VectorDims{},
                                                  blk_strides);
}

std::shared_ptr<MemoryDesc> MemoryDescUtils::makeDummyDesc(const MemoryDesc& desc, Dim dummyVal) {
    auto dummyShape = makeDummyShape(desc.getShape(), dummyVal);
    return desc.cloneWithNewDims(dummyShape.getStaticDims());
}

std::shared_ptr<MemoryDesc> MemoryDescUtils::makeEmptyDesc() {
    static auto emptyDesc = std::make_shared<EmptyMemoryDesc>();
    return emptyDesc;
}

std::shared_ptr<IMemory> MemoryDescUtils::makeEmptyMemory(const GraphContext::CPtr& context) {
    return std::make_shared<StaticMemory>(context->getEngine(), makeEmptyDesc(), nullptr);
}

Shape MemoryDescUtils::makeDummyShape(const Shape& shape, Dim dummyVal) {
    const auto& minDims = shape.getMinDims();
    const auto& maxDims = shape.getMaxDims();
    const auto& dims = shape.getDims();
    VectorDims dummyDims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        dummyDims[i] = dims[i] == Shape::UNDEFINED_DIM ? std::min(maxDims[i], std::max(minDims[i], dummyVal)) : dims[i];
    }
    return Shape(dummyDims);
}

Shape MemoryDescUtils::makeDummyShape(const Shape& shape, const VectorDims& dummyVals) {
    OPENVINO_ASSERT(shape.getRank() == dummyVals.size(),
                    "makeDummyShape(): dummyVals vector size and shape ranks mismatch");
    const auto& minDims = shape.getMinDims();
    const auto& maxDims = shape.getMaxDims();
    const auto& dims = shape.getDims();
    VectorDims dummyDims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        dummyDims[i] =
            dims[i] == Shape::UNDEFINED_DIM ? std::min(maxDims[i], std::max(minDims[i], dummyVals[i])) : dims[i];
    }
    return Shape(dummyDims);
}
}  // namespace ov::intel_cpu
