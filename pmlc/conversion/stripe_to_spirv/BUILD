# Copyright 2019 Intel Corporation.

package(default_visibility = ["//visibility:public"])

load("//bzl:plaidml.bzl", "plaidml_cc_library")
load("//vendor/mlir:mlir.bzl", "COPTS")

plaidml_cc_library(
    name = "stripe_to_spirv",
    srcs = [
        "convert_stripe_to_spirv.cc",
        "convert_stripe_to_spirv.h",
    ],
    copts = COPTS,
    deps = [
    	"//pmlc/conversion/stripe_to_affine",
        "//pmlc/dialect/stripe",
        "@mlir//:SPIRVIR",
    ],
    alwayslink=1,
)
