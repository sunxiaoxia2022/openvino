// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iremote_tensor.hpp"

#include <memory>

#include "dev/make_tensor.hpp"
#include "ie_blob.h"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

IRemoteTensor::~IRemoteTensor() = default;

/**
 * @brief Tensor what contains InferenceEngine::Blob inside
 * Blob owns the memory
 */
class BlobTensor : public ITensor {
    mutable element::Type m_type;
    mutable Shape m_shape;
    mutable Strides m_strides;

    void update_strides() {
        if (get_element_type().bitwidth() >= 8) {
            const auto& element_strides = blob->getTensorDesc().getBlockingDesc().getStrides();
            const size_t elem_size = get_element_type().size();
            m_strides.clear();
            m_strides.resize(element_strides.size());
            std::transform(element_strides.begin(),
                           element_strides.end(),
                           m_strides.begin(),
                           [&elem_size](size_t stride) {
                               return stride * elem_size;
                           });
        }
    }

public:
    std::shared_ptr<ie::Blob> blob;

    BlobTensor(const InferenceEngine::Blob::Ptr& blob) : blob{blob} {
        auto remote_impl = dynamic_cast<InferenceEngine::RemoteBlob*>(blob.get());
        OPENVINO_ASSERT(!remote_impl);
        OPENVINO_ASSERT(blob);
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
        update_strides();
    }

    const element::Type& get_element_type() const override {
        m_type = InferenceEngine::details::convertPrecision(blob->getTensorDesc().getPrecision());
        return m_type;
    }

    void set_shape(ov::Shape shape) override {
        blob->setShape({shape.begin(), shape.end()});
        update_strides();
    }

    const Shape& get_shape() const override {
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
        return m_shape;
    }

    const Strides& get_strides() const override {
        OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        get_element_type());
        return m_strides;
    }

    size_t get_size() const override {
        return blob->size();
    }

    size_t get_byte_size() const override {
        return blob->byteSize();
    }

    void* data(const element::Type& element_type) const override {
        OPENVINO_ASSERT(blob != nullptr, "Tensor was not initialized.");
#define TYPE_CHECK(TYPE) (dynamic_cast<const ie::TBlob<TYPE>*>(blob.get()) != nullptr)
        auto host_accesable_implementation = TYPE_CHECK(bool) || TYPE_CHECK(int8_t) || TYPE_CHECK(uint8_t) ||
                                             TYPE_CHECK(int16_t) || TYPE_CHECK(uint16_t) || TYPE_CHECK(int32_t) ||
                                             TYPE_CHECK(uint32_t) || TYPE_CHECK(int64_t) || TYPE_CHECK(uint64_t) ||
                                             TYPE_CHECK(float) || TYPE_CHECK(double);
#undef TYPE_CHECK
        OPENVINO_ASSERT(host_accesable_implementation,
                        "Tensor implementation type dose not contains host accessable data");
        if (element_type != element::undefined && element_type.is_static()) {
            OPENVINO_ASSERT(element_type == get_element_type(),
                            "Tensor data with element type ",
                            get_element_type(),
                            ", is not representable as pointer to ",
                            element_type);
        }
        // since we don't use byte offsets, we need to explicitly multiply by element_size
        auto byte_offset = blob->getTensorDesc().getBlockingDesc().getOffsetPadding() * get_element_type().size();
        OPENVINO_ASSERT((get_element_type().bitwidth() >= 8) || (byte_offset == 0),
                        "ROI access for types with bitwidths less then 8 bit is not implemented. Tensor type: ",
                        get_element_type());
        return byte_offset + InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<uint8_t*>();
    }
};

/**
 * @brief Tensor what contains InferenceEngine::RemoteBlob inside
 * Blob owns the memory
 */
class RemoteBlobTensor : public IRemoteTensor {
    mutable element::Type m_type;
    mutable Shape m_shape;
    mutable Strides m_strides;
    mutable ov::AnyMap m_properties;
    mutable std::string m_dev_name;

public:
    std::shared_ptr<ie::RemoteBlob> blob;

    RemoteBlobTensor(const InferenceEngine::RemoteBlob::Ptr& blob) : blob{blob} {
        OPENVINO_ASSERT(blob);
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
    }

    const element::Type& get_element_type() const override {
        m_type = InferenceEngine::details::convertPrecision(blob->getTensorDesc().getPrecision());
        return m_type;
    }

    void set_shape(ov::Shape shape) override {
        blob->setShape({shape.begin(), shape.end()});
    }

    const Shape& get_shape() const override {
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
        return m_shape;
    }

    const Strides& get_strides() const override {
        OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        get_element_type());
        const auto& element_strides = blob->getTensorDesc().getBlockingDesc().getStrides();
        const size_t elem_size = get_element_type().size();
        m_strides.clear();
        m_strides.resize(element_strides.size());
        std::transform(element_strides.begin(), element_strides.end(), m_strides.begin(), [&elem_size](size_t stride) {
            return stride * elem_size;
        });
        return m_strides;
    }

    size_t get_size() const override {
        return blob->size();
    }

    size_t get_byte_size() const override {
        return blob->byteSize();
    }

    const AnyMap& get_properties() const override {
        m_properties = blob->getParams();
        return m_properties;
    }

    const std::string& get_device_name() const override {
        m_dev_name = blob->getDeviceName();
        return m_dev_name;
    }
};

/**
 * @brief Create InferenceEngine::RemoteBlob from the Tensor
 */
class TensorRemoteBlob : public ie::RemoteBlob {
public:
    TensorRemoteBlob(const std::shared_ptr<ITensor>& tensor)
        : ie::RemoteBlob{ie::TensorDesc{ie::details::convertPrecision(tensor->get_element_type()),
                                        tensor->get_shape(),
                                        ie::TensorDesc::getLayoutByRank(tensor->get_shape().size())}},
          tensor{std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor)} {
        OPENVINO_ASSERT(this->tensor);
    }
    AnyMap getParams() const override {
        return tensor->get_properties();
    }
    std::string getDeviceName() const noexcept override {
        try {
            return tensor->get_device_name();
        } catch (...) {
            return {};
        }
    }
    std::shared_ptr<ie::RemoteContext> getContext() const noexcept override {
        return {};
    }

    void allocate() noexcept override {}
    bool deallocate() noexcept override {
        return true;
    }
    ie::LockedMemory<void> buffer() noexcept override {
        return {nullptr, nullptr, 0};
    }
    ie::LockedMemory<const void> cbuffer() const noexcept override {
        return {nullptr, nullptr, 0};
    }
    ie::LockedMemory<void> rwmap() noexcept override {
        return {nullptr, nullptr, 0};
    }
    ie::LockedMemory<const void> rmap() const noexcept override {
        return {nullptr, nullptr, 0};
    }
    ie::LockedMemory<void> wmap() noexcept override {
        return {nullptr, nullptr, 0};
    }
    const std::shared_ptr<ie::IAllocator>& getAllocator() const noexcept override {
        return m_allocator;
    }
    void* getHandle() const noexcept override {
        return nullptr;
    }

    std::shared_ptr<IRemoteTensor> tensor;

private:
    std::shared_ptr<ie::IAllocator> m_allocator;
};

/**
 * @brief Create InferenceEngine::TBlob<T> from the tensor
 *
 * @tparam T Blob data type
 */
template <typename T>
class TensorMemoryBlob : public ie::TBlob<T> {
public:
    ~TensorMemoryBlob() override = default;
    explicit TensorMemoryBlob(const std::shared_ptr<ITensor>& tensor_) try : ie
        ::TBlob<T>{[&] {
                       auto element_type = tensor_->get_element_type();
                       auto shape = tensor_->get_shape();
                       ie::SizeVector blk_order(shape.size());
                       std::iota(blk_order.begin(), blk_order.end(), 0);
                       ie::SizeVector dim_offset(shape.size(), 0);
                       ie::SizeVector blk_strides;
                       auto byte_strides = element_type.bitwidth() >= 8 ? tensor_->get_strides() : Strides{};
                       if (byte_strides.empty()) {
                           blk_strides = ov::row_major_strides(shape);
                       } else {
                           blk_strides.resize(byte_strides.size());
                           std::transform(byte_strides.begin(),
                                          byte_strides.end(),
                                          blk_strides.begin(),
                                          [&element_type](size_t byte_stride) {
                                              OPENVINO_ASSERT(byte_stride % element_type.size() == 0,
                                                              "Limitation: Stride in bytes ",
                                                              byte_stride,
                                                              " should be divisible by size of element ",
                                                              element_type.size());
                                              return byte_stride / element_type.size();
                                          });
                       }
                       return ie::TensorDesc{ie::details::convertPrecision(element_type),
                                             shape,
                                             ie::BlockingDesc{shape, blk_order, 0, dim_offset, blk_strides}};
                   }(),
                   static_cast<T*>(tensor_->data()),
                   tensor_->get_byte_size()},
            tensor{tensor_} {
            OPENVINO_ASSERT(!std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor));
        }
    catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    }

    void setShape(const ie::SizeVector& dims) override {
        tensor->set_shape(dims);
        ie::TBlob<T>::setShape(dims);
    }

    std::shared_ptr<ITensor> tensor;
};

std::shared_ptr<ITensor> make_tensor(const std::shared_ptr<ie::Blob>& blob) {
#define ELSE_IF(type)                                                                \
    else if (auto tblob = dynamic_cast<const TensorMemoryBlob<type>*>(blob.get())) { \
        return tblob->tensor;                                                        \
    }
    if (blob == nullptr) {
        return {};
    } else if (auto remote_blob = std::dynamic_pointer_cast<TensorRemoteBlob>(blob)) {
        return remote_blob->tensor;
    } else if (auto remote_blob = std::dynamic_pointer_cast<InferenceEngine::RemoteBlob>(blob)) {
        return std::make_shared<RemoteBlobTensor>(remote_blob);
    }
    ELSE_IF(float)
    ELSE_IF(double)
    ELSE_IF(int8_t)
    ELSE_IF(int8_t)
    ELSE_IF(int16_t)
    ELSE_IF(int32_t)
    ELSE_IF(int64_t)
    ELSE_IF(uint8_t)
    ELSE_IF(uint8_t)
    ELSE_IF(uint16_t)
    ELSE_IF(uint32_t)
    ELSE_IF(uint64_t)
    ELSE_IF(int8_t)
    ELSE_IF(bool) else {
        return std::make_shared<BlobTensor>(blob);
    }
#undef IF
}

ie::Blob::Ptr tensor_to_blob(const std::shared_ptr<ITensor>& tensor) {
    if (tensor == nullptr) {
        return {};
    } else if (auto blob_tensor = std::dynamic_pointer_cast<BlobTensor>(tensor)) {
        return blob_tensor->blob;
    } else if (auto blob_tensor = std::dynamic_pointer_cast<RemoteBlobTensor>(tensor)) {
        return blob_tensor->blob;
    } else if (auto blob_tensor = dynamic_cast<const BlobTensor*>(tensor.get())) {
        return blob_tensor->blob;
    } else if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor)) {
        return std::make_shared<TensorRemoteBlob>(tensor);
    } else {
#define CASE(precision, T)   \
    case element::precision: \
        return std::make_shared<TensorMemoryBlob<T>>(tensor);
        switch (tensor->get_element_type()) {
            CASE(f32, float);
            CASE(f64, double);
            CASE(i4, int8_t);
            CASE(i8, int8_t);
            CASE(i16, int16_t);
            CASE(i32, int32_t);
            CASE(i64, int64_t);
            CASE(u4, uint8_t);
            CASE(u8, uint8_t);
            CASE(u16, uint16_t);
            CASE(u32, uint32_t);
            CASE(u64, uint64_t);
            CASE(u1, int8_t);
            CASE(boolean, bool);
        case element::f16:
            return std::make_shared<TensorMemoryBlob<int16_t>>(tensor);
        case element::bf16:
            return std::make_shared<TensorMemoryBlob<int16_t>>(tensor);
        default:
            OPENVINO_THROW("Unsupported element type");
        }
#undef CASE
    }
    OPENVINO_THROW("Cannot convert tensor to blob!");
}
}  // namespace ov
