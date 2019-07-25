/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file np_unique_op.cc
 */

#include "./np_unique_op-inl.h"

namespace mxnet {
namespace op {



template<>
void NumpyUniqueForward<cpu>(const nnvm::NodeAttrs& attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  // CHECK_EQ(outputs.size(), 1U);
  CHECK(req[0] == kWriteTo || req[0] == kWriteInplace);
  const NumpyUniqueParam& param = nnvm::get<NumpyUniqueParam>(attrs.parsed);
  if (inputs[0].shape().ndim() == 0) {

  } else if (inputs[0].shape().Size() == 0) {

  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      DType* input_data = inputs[0].data().dptr<DType>();
      size_t input_size = inputs[0].shape().Size();
      if (!param.axis.has_value()) {
        if (param.return_index || param.return_inverse || param.return_counts) {
          // argsort, result in perm
          std::vector<dim_t> perm(input_size);
          std::iota(perm.begin(), perm.end(), 0);
          std::sort(perm.begin(), perm.end(), [&input_data](size_t i1, size_t i2) {return input_data[i1] < input_data[i2];});
          // sorted data in aux
          std::vector<DType> aux(input_size);
          for (size_t i = 0; i < input_size; ++i) {
            aux[i] = input_data[perm[i]];
          }
          // calculate unique mask
          std::vector<int32_t> mask(input_size);
          mask[0] = 1;
          for (size_t i = 1; i < input_size; ++i) {
            mask[i] = (aux[i] == aux[i - 1]) ? 0 : 1;
          }
          // Calculate prefix sum
          std::vector<int32_t> prefix_sum(input_size, 0);
          size_t valid_num = 0;
          for (size_t i = 0; i < input_size; i++) {
            prefix_sum[i] = (i == 0) ? 0 : prefix_sum[i - 1];
            prefix_sum[i] += (mask[i]) ? 1 : 0;
          }
          valid_num = prefix_sum[input_size - 1];
          // set the output shape forcefully
          mxnet::TShape s(1, valid_num);
          const_cast<NDArray &>(outputs[0]).Init(s);
          // launch kernal to obtain unique array, reuse boolean_mask kernel
          mshadow::Stream<cpu> *stream = ctx.get_stream<cpu>();
          mxnet_op::Kernel<BooleanMaskForwardCPUKernel, cpu>::Launch(
            stream, input_size, outputs[0].data().dptr<DType>(), aux.data(),
            prefix_sum.data(), 1);
          // handle other optional outputs
          int output_flag = 0;
          if (param.return_index) {
            output_flag += 1;
            const_cast<NDArray &>(outputs[output_flag]).Init(s);
            dim_t* unique_indices = outputs[output_flag].data().dptr<dim_t>();
            // reuse boolean_mask kernel
            mxnet_op::Kernel<BooleanMaskForwardCPUKernel, cpu>::Launch(
              stream, input_size, unique_indices, perm.data(),
              prefix_sum.data(), 1);
          }
          if (param.return_inverse) {
            output_flag += 1;
            const_cast<NDArray &>(outputs[output_flag]).Init(mxnet::TShape(1, input_size));
            dim_t* unique_inverse = outputs[output_flag].data().dptr<dim_t>();
            // TODO by Guangtai Huang, write by kernel
            for (size_t i = 0; i < input_size; ++i) {
              unique_inverse[perm[i]] = prefix_sum[i] - 1;
            }
          }
          if (param.return_counts) {
            output_flag += 1;
            std::vector<int32_t> idx(valid_num + 1);
            auto iter = idx.begin();
            for (size_t i = 0; i < input_size; ++i) {
              if (mask[i]) {
                *iter = i;
                ++iter;
              }
            }
            *iter = input_size;
            const_cast<NDArray &>(outputs[output_flag]).Init(s);
            dim_t* unique_counts = outputs[output_flag].data().dptr<dim_t>();
            // TODO by Guangtai Huang, write by kernel
            for (size_t i = 0; i < valid_num; ++i) {
              unique_counts[i] = idx[i + 1] - idx[i];
            }
          }
        } else {
          std::set<DType> set(input_data, input_data + input_size);
          mxnet::TShape s(1, set.size());
          const_cast<NDArray &>(outputs[0]).Init(s);
          std::copy(set.begin(), set.end(), outputs[0].data().dptr<DType>());
        }
      } else {

      }
    });
  }
}

DMLC_REGISTER_PARAMETER(NumpyUniqueParam);

NNVM_REGISTER_OP(_npi_unique)
.set_attr_parser(ParamParser<NumpyUniqueParam>)
.set_num_inputs(1)
.set_num_outputs([](const NodeAttrs& attrs) {
    const NumpyUniqueParam& param = nnvm::get<NumpyUniqueParam>(attrs.parsed);
    int output_num = 1;
    if (param.return_index) output_num += 1;
    if (param.return_inverse) output_num += 1;
    if (param.return_counts) output_num += 1;
    return output_num;
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferType>("FInferType", NumpyUniqueType)
.set_attr<FComputeEx>("FComputeEx<cpu>", NumpyUniqueForward<cpu>)
.set_attr<FInferStorageType>("FInferStorageType", NumpyUniqueStorageType)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("data", "NDArray-or-Symbol", "The input array")
.add_arguments(NumpyUniqueParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
