// Copyright 2019 Intel Corporation.

#include "pmlc/dialect/tile/gradient.h"

namespace pmlc {
namespace dialect {
namespace tile {

// TODO: I'm unclear if GlobalContext::get() is safe/sensible here, or if Gradient should be otherwise connected to a
// TileBuilder

// TODO

struct UseInfo {
  mlir::Value* val;
  size_t idx;
};

class Gradient {
 public:
  explicit Gradient(const mlir::Value* loss) : uses_(loss) {
    IVLOG(4, "Gradient::Gradient> loss: " << loss);
    grads_[loss.get()] = GlobalContext::get()->MakeScalarConstantOp(1.0);
  }

  mlir::Value* GetDerivative(const mlir::Value* val) {
    IVLOG(4, "Gradient::GetDerivative> " << val);
    auto it = grads_.find(val);
    if (it != grads_.end()) {
      IVLOG(5, "  returning: " << it->second);
      return it->second;
    }
    mlir::Value* total;
    for (const auto& use : uses_.uses(val)) {
      mlir::Value* dop;
      auto dout = GetDerivative(use.val);
      // TODO: From here
      if (auto grad_override_expr = std::dynamic_pointer_cast<GradOverrideExpr>(use.expr)) {
        // A gradient override replaces all the derivatives, so set total and exit the loop
        total = DeriveOverride(dout, grad_override_expr, use.idx);
        break;
      }
      if (auto call_expr = std::dynamic_pointer_cast<CallExpr>(use.expr)) {
        dop = DeriveCall(dout, call_expr, use.idx);
      } else if (auto cion_expr = std::dynamic_pointer_cast<ContractionExpr>(use.expr)) {
        dop = DeriveContraction(dout, cion_expr, use.idx);
      } else {
        throw std::runtime_error("Invalid operation type in Gradient::GetDerivative");
      }
      if (!total) {
        total = dop;
      } else {
        total = MakeCall("add", {total, dop});
      }
    }
    if (!total) {
      total = std::make_shared<FloatConst>(0.0);
    } else if (total->shape.dims.size()) {
      total = MakeCall("simple_reduce", {total, expr});
    }
    IVLOG(4, "  Gradient::GetDerivative, final result -> " << total);
    seen_.emplace(expr.get(), total);
    return total;
  }

 private:
  ComputeUses uses_;
  std::map<const mlir::Value*, mlir::Value*> grads_;
}

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
