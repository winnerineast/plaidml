// Copyright 2019 Intel Corporation

#pragma once

#include <memory>

namespace mlir {

class FuncOp;
class MLIRContext;
template <typename T>
class OpPassBase;
using FunctionPassBase = OpPassBase<FuncOp>;
class OwningRewritePatternList;

/// Creates a pass to convert Stripe dialect to spirv dialect.
std::unique_ptr<FunctionPassBase> createStripeToSPIRVLoweringPass();

}  // namespace mlir
