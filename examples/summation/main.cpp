#include <iostream>
#include <cugrad.hpp>

auto main() -> int
{
    auto a = cugrad::create_tensor({ 10 });
    auto b = cugrad::sum(a);

    a->set(2);

    auto result = b->compute()->cpu();

    std::printf("a.shape = %s\n", cugrad::shape_to_str(a->shape()).c_str());
    std::printf("b.shape = %s\n", cugrad::shape_to_str(b->shape()).c_str());
    std::printf("result = %f\n", result[0]);

    auto pa = b->grad()->cpu(a)[0]; // 0 ... 9

    std::printf("partial with respect to a = %f\n", pa);
}