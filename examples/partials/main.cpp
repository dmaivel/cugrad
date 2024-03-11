#include <iostream>
#include <cugrad.hpp>

auto main() -> int
{
    auto a = cugrad::create_tensor({ 1 });
    auto b = cugrad::create_tensor({ 1 });

    auto c = cugrad::add(a, b);
    auto d = cugrad::mul(a, c);

    a->set(4);
    b->set(3);

    auto result = d->compute()->cpu()[0];

    std::printf("result = %f\n", result);

    d->grad();
    auto pa = d->cpu(a)[0];
    auto pb = d->cpu(b)[0];
    auto pc = d->cpu(c)[0];

    std::printf("partial with respect to a = %f\n", pa);
    std::printf("partial with respect to b = %f\n", pb);
    std::printf("partial with respect to c = %f\n", pc);
}