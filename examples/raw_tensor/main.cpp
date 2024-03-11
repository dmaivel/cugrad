#include <iostream>
#include <cugrad.hpp>

auto main() -> int
{
    /*
     * nothing here runs on the GPU, this is a test of the raw_tensor class
     * which will hold data returned from the GPU.
     */

    auto tensor = cugrad::raw_tensor({ 3, 3 });

    for (int i = 0; i < 3 * 3; i++)
        tensor[i] = float(i);

    tensor[8] = 123;

    std::printf("tensor[0] = %f\n", tensor[0]);
    std::printf("tensor[8] = %f\n", tensor[8]);
    std::printf("tensor[0, 0] = %f\n", tensor({ 0, 0 }));
    std::printf("tensor[1, 0] = %f\n", tensor({ 1, 0 }));
    std::printf("tensor[0, 1] = %f\n", tensor({ 0, 1 }));
    std::printf("tensor[1, 1] = %f\n", tensor({ 1, 1 }));
    std::printf("tensor[2, 2] = %f\n", tensor({ 2, 2 }));
}