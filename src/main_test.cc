#include <iostream>
#include "triton_op/infinitensor_triton.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/embed.h>
namespace py = pybind11;

using namespace infini;



int main() {
    py::scoped_interpreter guard{};

    std::cout<<"triton op"<<std::endl;
    print_test_add();

    std::cout<<"ninetoothed op"<<std::endl;
    print_test_add_ninetoothed();
    return 0;
}

