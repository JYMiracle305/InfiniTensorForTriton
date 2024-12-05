#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <iostream>
#include <pybind11/embed.h>
#include "triton_op/infinitensor_triton.h"
namespace py = pybind11;

namespace infini {
void whereKernel_py(const float *inputX, const float *inputY,
                 const uint8_t *condition, float *output, int nDims,
                 int outputsize, SmallArray inputXShape, SmallArray inputYShape,
                 SmallArray conditionShape, SmallArray outputShape, int xSize,
                 int ySize, int cSize) {

    py::scoped_interpreter guard{};

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        // 构建 Python 模块的路径
        std::string op_module_path = std::string(cwd) + "/python/tritonOp";
        std::cout << "op_module_path: " << op_module_path << std::endl;

        const char* home_dir = std::getenv("HOME");
        std::string python_module_path;
        if (home_dir != nullptr) {
            python_module_path = std::string(home_dir) + "/.local/lib/python3.10/site-packages";
            std::cout << "python_module_path: " << python_module_path << std::endl;
        } else {
            std::cerr << "HOME environment variable is not set." << std::endl;
        }

        // 导入 sys 模块并添加模块路径
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, py::str(op_module_path));
        sys.attr("path").attr("insert")(0, py::str(python_module_path));

        // 导入 vector_add 模块
        py::module_ main_module = py::module_::import("where_op");
        py::object whereKernel = main_module.attr("whereKernel");

        // 调用 Python 的 vector_add 函数
        whereKernel(inputX, inputY, condition, output, nDims, outputsize,
            inputXShape, inputYShape, conditionShape, outputShape, xSize, ySize, cSize);

        std::cout << "Result: " << *output << std::endl;
    } else {
        std::cerr << "Error getting current working directory: " << strerror(errno) << std::endl;
    }
}

}