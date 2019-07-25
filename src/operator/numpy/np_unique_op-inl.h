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
 * \file np_unique_op-inl.h
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_INL_H_

#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <vector>
#include <set>
#include <string>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../contrib/boolean_mask-inl.h"

namespace mxnet {
namespace op {

struct NumpyUniqueParam : public dmlc::Parameter<NumpyUniqueParam> {
  bool return_index, return_inverse, return_counts;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(NumpyUniqueParam) {
    DMLC_DECLARE_FIELD(return_index)
    .set_default(false)
    .describe("If true, return the indices of the input.");
    DMLC_DECLARE_FIELD(return_inverse)
    .set_default(false)
    .describe("If true, return the indices of the input.");
    DMLC_DECLARE_FIELD(return_counts)
    .set_default(false)
    .describe("If true, return the number of times each unique item appears in input.");
    DMLC_DECLARE_FIELD(axis)
    .set_default(dmlc::optional<int>())
    .describe("An integer that represents the axis to operator on.");
  }
};

inline bool NumpyUniqueType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  // CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  for (size_t i = 1; i < out_attrs->size(); ++i) {
    TYPE_ASSIGN_CHECK(*out_attrs, i, mshadow::kInt64);
  }
  return out_attrs->at(0) != -1;
}

inline bool NumpyUniqueStorageType(const nnvm::NodeAttrs& attrs,
                            const int dev_mask,
                            DispatchMode* dispatch_mode,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  // CHECK_EQ(out_attrs->size(), 1U);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

template<typename xpu>
void NumpyUnique1D(const nnvm::NodeAttrs& attrs,
                   const OpContext &ctx,
                   const std::vector<NDArray> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<NDArray> &outputs);

template<typename xpu>
void NumpyUniqueForward(const nnvm::NodeAttrs& attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_INL_H_
