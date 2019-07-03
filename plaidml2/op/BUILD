# Copyright 2019 Intel Corporation.

load(
    "//bzl:plaidml.bzl",
    "plaidml_cc_library",
    "plaidml_cc_test",
    "plaidml_py_library",
)

exports_files([
    "ffi.h",
])

plaidml_cc_library(
    name = "op",
    srcs = [
        "ffi.cc",
        "ops.cc",
        "ops.h",
    ],
    hdrs = [
        "ffi.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//plaidml2/core",
        "//plaidml2/edsl",
        "@boost",
    ],
    alwayslink = 1,
)

plaidml_cc_library(
    name = "api",
    hdrs = [
        "ffi.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml2/core:api",
        "//plaidml2/edsl:api",
    ],
)

plaidml_py_library(
    name = "py",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//plaidml2:py",
        "//plaidml2/edsl:py",
    ],
)