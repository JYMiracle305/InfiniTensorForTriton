import numpy as np


# void whereKernel_py(const float *inputX, const float *inputY,
#                  const uint8_t *condition, float *output, int nDims,
#                  int outputsize, SmallArray inputXShape, SmallArray inputYShape,
#                  SmallArray conditionShape, SmallArray outputShape, int xSize,
#                  int ySize, int cSize);

def whereKernel(inputX, inputY, condition, output, nDims, outputsize,
                inputXShape, inputYShape, conditionShape, outputShape, xSize, ySize, cSize):
    print("---------", type(inputX), type(inputY))
    # 将输入转换为numpy数组
    inputX_np = np.array(inputX, dtype=np.float32)
    inputY_np = np.array(inputY, dtype=np.float32)
    condition_np = np.array(condition, dtype=np.uint8)

    # 确保输入数组的形状与给定的形状匹配
    inputX_np = inputX_np.reshape(inputXShape)
    inputY_np = inputY_np.reshape(inputYShape)
    condition_np = condition_np.reshape(conditionShape)

    # # 广播输入数组以匹配输出形状
    inputX_np = np.broadcast_to(inputX_np, outputShape)
    inputY_np = np.broadcast_to(inputY_np, outputShape)
    condition_np = np.broadcast_to(condition_np, outputShape)

    # 执行条件选择操作
    output_np = np.where(condition_np, inputX_np, inputY_np)

    # 将结果复制到输出数组
    output[:] = output_np.flatten()

if __name__ == "__main__":
    inputX = np.array([[1.0, 2.0],[1.0, 2.0]], dtype=np.float32)
    inputY = np.array([[3.0, 4.0],[5.0, 6.0]], dtype=np.float32)
    condition = np.array([[True, False],[True, True]], dtype=np.uint8)

    # 输出数组
    output = np.empty(4, dtype=np.float32)

    # 调用函数
    whereKernel(inputX, inputY, condition, output, 1, 2, (2,2), (2,2), (2,2), (2,2), 1, 1, 1)

    # 输出结果
    print(output)  # [1. 4.]