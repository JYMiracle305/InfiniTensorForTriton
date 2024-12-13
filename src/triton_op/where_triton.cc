#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h> // For py::scoped_interpreter
#include <iostream>
#include <cstring>
#include <unistd.h> // For getcwd

namespace py = pybind11;
namespace infini {


// 将 C++ 的 float* 转换为 numpy 数组
py::array_t<float> convert_to_numpy_float(const float* data, size_t size) {
    if (data == nullptr) {
        throw std::runtime_error("Input data pointer is null.");
    }

    auto capsule = py::capsule(data, [](void *ptr) {
        // 如果需要释放内存，可以在这里实现
        // delete[] static_cast<float*>(ptr);
    });

    return py::array_t<float>(
        {size},          // 形状 (1D 数组)
        {sizeof(float)}, // 步长 (每个元素的字节数)
        data,            // 数据指针
        capsule          // 内存管理
    );
}

// 将 C++ 的 uint8_t* 转换为 numpy 数组
py::array_t<uint8_t> convert_to_numpy_uint8(const uint8_t* data, size_t size) {
    if (data == nullptr) {
        throw std::runtime_error("Condition data pointer is null.");
    }

    auto capsule = py::capsule(data, [](void *ptr) {
        // 如果需要释放内存，可以在这里实现
        // delete[] static_cast<uint8_t*>(ptr);
    });

    return py::array_t<uint8_t>(
        {size},          // 形状 (1D 数组)
        {sizeof(uint8_t)}, // 步长 (每个元素的字节数)
        data,              // 数据指针
        capsule            // 内存管理
    );
}

// 将 SmallArray 转换为 numpy 数组
py::array_t<int> convert_to_numpy_int_array(const SmallArray& array) {
    return py::array_t<int>(
        {SMALL_ARRAY_SIZE}, // 形状 (1D 数组)
        {sizeof(int)},      // 步长 (每个元素的字节数)
        array.data          // 数据指针
    );
}

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

        // 导入 Python 模块
        py::module_ main_module = py::module_::import("where_op");
        py::object whereKernel = main_module.attr("whereKernel");

        std::cout << "inputX pointer: " << (void*)inputX << ", size: " << xSize << std::endl;
        std::cout << "inputY pointer: " << (void*)inputY << ", size: " << ySize << std::endl;
        
        // 将 C++ 指针转换为 numpy 数组
        py::array_t<float> inputX_np = convert_to_numpy_float(inputX, xSize);
        py::array_t<float> inputY_np = convert_to_numpy_float(inputY, ySize);
        py::array_t<uint8_t> condition_np = convert_to_numpy_uint8(condition, cSize);

        // 将 SmallArray 转换为 numpy 数组
        py::array_t<int> inputXShape_np = convert_to_numpy_int_array(inputXShape);
        py::array_t<int> inputYShape_np = convert_to_numpy_int_array(inputYShape);
        py::array_t<int> conditionShape_np = convert_to_numpy_int_array(conditionShape);
        py::array_t<int> outputShape_np = convert_to_numpy_int_array(outputShape);

        // 调用 Python 的 whereKernel 函数
        whereKernel(inputX_np, inputY_np, condition_np, output, nDims, outputsize,
                    inputXShape_np, inputYShape_np, conditionShape_np, outputShape_np,
                    xSize, ySize, cSize);

        std::cout << "Result: " << *output << std::endl;
    } else {
        std::cerr << "Error getting current working directory: " << strerror(errno) << std::endl;
    }
}

void whereKernel_test(const float *inputX, const float *inputY, float *output, int xSize,
                 int ySize)
{
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

        // 导入 Python 模块
        py::module_ main_module = py::module_::import("where_op");
        py::object whereKernel = main_module.attr("whereKernel");

        std::cout << "inputX pointer: " << (void*)inputX << ", size: " << xSize << std::endl;
        std::cout << "inputY pointer: " << (void*)inputY << ", size: " << ySize << std::endl;
        
        // 将 C++ 指针转换为 numpy 数组
        py::array_t<float> inputX_np = convert_to_numpy_float(inputX, xSize);
        py::array_t<float> inputY_np = convert_to_numpy_float(inputY, ySize);

        whereKernel(inputX_np, inputY_np, output, xSize, ySize);

        std::cout << "Result: " << *output << std::endl;
    } else {
        std::cerr << "Error getting current working directory: " << strerror(errno) << std::endl;
    }
}
} // namespace infini