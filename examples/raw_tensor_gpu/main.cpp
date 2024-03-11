#include <iostream>
#include <memory>

#include <cugrad.hpp>

auto print_tensor(const std::string &name, const cugrad::raw_tensor &tensor, bool no_extra_newline = false) -> void
{
    std::printf("%s = [\n", name.c_str());
    for (int y = 0; y < 3; y++) {
        std::printf("\t[ ");
        for (int x = 0; x < 3; x++)
            std::printf("%5.1f ", tensor({ x, y }));

        std::printf("]\n");
    }
    std::printf("]\n");

    // just to make formatting to terminal a little nicer
    if (!no_extra_newline)
        std::puts("");
}

auto main() -> int
{
    auto a = cugrad::create_tensor({ 3, 3 });
    auto b = cugrad::create_tensor({ 3, 3 });

    // f(a, b) = (a + b) * (b - a)
    auto f = [](std::shared_ptr<cugrad::tensor> a, std::shared_ptr<cugrad::tensor> b) {
        return cugrad::mul(cugrad::add(a, b), cugrad::sub(b, a));
    };

    auto y = f(a, b);

    // set contents of a and b
    {
        auto raw_contents_a = cugrad::raw_tensor({ 3, 3 });
        auto raw_contents_b = cugrad::raw_tensor({ 3, 3 });

        for (int i = 0; i < 3 * 3; i++) {
            raw_contents_a[i] = float(i + 1);
            raw_contents_b[i] = float(9 - i);
        }

        a->set(raw_contents_a);
        b->set(raw_contents_b);
    }

    // compute and store result
    cugrad::raw_tensor c = y->compute()->cpu();

    // store result c in b
    b->set(c);

    // recalculate
    cugrad::raw_tensor d = y->compute()->cpu();

    // print results for c
    print_tensor("c", c);

    // print results for d
    print_tensor("d", d);

    // calculate gradients with respect to
    cugrad::raw_tensor d_grad_a = y->grad()->cpu(a);
    cugrad::raw_tensor d_grad_b = y->cpu(b);

    // print results for d_grad with respect to a
    print_tensor("d_grad wrt a", d_grad_a);

    // print results for d_grad with respect to b
    print_tensor("d_grad wrt b", d_grad_b, true);
}