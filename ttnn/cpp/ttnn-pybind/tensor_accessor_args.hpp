// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace ttnn::tensor_accessor_args {

namespace py = pybind11;
void py_module_types(py::module& module);
void py_module(py::module& module);

}  // namespace ttnn::tensor_accessor_args
