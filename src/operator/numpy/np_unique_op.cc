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
    if (!param.axis.has_value()) {
      NumpyUniqueCPUNoneAxisImpl(param, ctx, inputs, req, outputs);
    } else {
      NumpyUniqueCPUImpl(param, ctx, inputs, req, outputs);
    }
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
