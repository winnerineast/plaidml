#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LowerAffine.h"
#include "llvm/ADT/Sequence.h"

#include "pmlc/conversion/stripe_to_affine/convert_stripe_to_affine.h"
#include "pmlc/conversion/stripe_to_spirv/convert_stripe_to_spirv.h"

namespace mlir {

struct StripeToSPIRVLoweringPass : public ModulePass<StripeToSPIRVLoweringPass> {
  void runOnModule() final;
};

void StripeToSPIRVLoweringPass::runOnModule() {
  ConversionTarget target(getContext());
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  SPIRVTypeConverter typeConverter();
  // TODO fix typeconverter to apply full conversion

  OwningRewritePatternList patterns;
  pmlc::conversion::stripe_to_affine::populateStripeToAffineConversionPatterns(patterns, &getContext());
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStandardToSPIRVPatterns(&getContext(), patterns);

  auto module = getModule();
  if (failed(applyFullConversion(module, target, patterns, &typeConverter))) signalPassFailure();
}

std::unique_ptr<mlir::Pass> createStripeToSPIRVLoweringPass() { return std::make_unique<StripeToSPIRVLoweringPass>(); }

}  // namespace mlir
