#include <iostream>
#include <cugrad.hpp>

auto main() -> int
{
    constexpr int input_size = 50;
    constexpr int output_size = 10;
    constexpr float lrate = 1e-3;

    auto x = cugrad::create_tensor({ input_size });
    auto y_true = cugrad::create_tensor({ output_size });
    auto weights = cugrad::create_tensor({ input_size, output_size });
    auto rate = cugrad::create_tensor({ input_size, output_size });
    auto gpu_gradient = cugrad::create_tensor({ input_size, output_size });

    x->set_rand(0.f);
    y_true->set_rand(0.f);
    weights->set_rand(0.f);
    rate->set(lrate);

    auto y_pred = cugrad::matmul(x, weights);
    auto loss = cugrad::mul(cugrad::sub(y_true, y_pred), cugrad::sub(y_true, y_pred));

    for (int i = 0; i < 100; i++) {
        // get loss value
        auto loss_val = cugrad::sum(loss)->compute()->cpu()[0];

        // tranfer gradient of loss with respect to weights back to gpu
        gpu_gradient->set(loss->grad()->cpu(weights));

        // update weights (weights - (rate * grad[weights]))
        weights->set(cugrad::sub(weights, cugrad::mul(rate, gpu_gradient))->compute()->cpu());

        // print loss
        std::printf("loss[%d] = %f\n", i, loss_val);
    }
}