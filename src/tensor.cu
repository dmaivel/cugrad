#include <cugrad.hpp>

#include <random>
#include <functional>

auto cugrad::tensor::set(const float v) const -> void
{
    auto host_array = std::make_unique<float[]>(n_elems);

    for (int i = 0; i < n_elems; i++)
        host_array[i] = v;

    cudaMemcpy(buffer, host_array.get(), alloc_size, cudaMemcpyHostToDevice);
}

auto cugrad::tensor::set(const cugrad::raw_tensor &tensor) const -> void
{
    CUGRAD_ASSERT(dims == tensor.shape(), "tensor has shape %s while raw_tensor has shape %s\n", shape_to_str(dims).c_str(), shape_to_str(tensor.shape()).c_str());

    auto host_array = std::make_unique<float[]>(n_elems);

    for (int i = 0; i < n_elems; i++)
        host_array[i] = tensor[i];

    cudaMemcpy(buffer, host_array.get(), alloc_size, cudaMemcpyHostToDevice);
}

auto cugrad::tensor::set_rand(const float min, const float max) const -> void
{
    auto host_array = std::make_unique<float[]>(n_elems);
    auto rng = std::bind(std::uniform_real_distribution<float>(min, max), std::mt19937(std::rand()));

    for (int i = 0; i < n_elems; i++)
        host_array[i] = rng();

    cudaMemcpy(buffer, host_array.get(), alloc_size, cudaMemcpyHostToDevice);
}

auto cugrad::tensor::cpu() const -> cugrad::raw_tensor 
{
    auto host_array = std::make_unique<float[]>(n_elems);

    cudaMemcpy(host_array.get(), buffer, alloc_size, cudaMemcpyDeviceToHost);

    return cugrad::raw_tensor(dims, n_elems, std::move(host_array));
}

auto cugrad::tensor::cpu(std::shared_ptr<cugrad::tensor> wrt) const -> cugrad::raw_tensor 
{
    auto host_array = std::make_unique<float[]>(wrt->n_elems);
    
    cudaMemcpy(host_array.get(), wrt->grad_buffer, wrt->alloc_size, cudaMemcpyDeviceToHost);

    return cugrad::raw_tensor(wrt->dims, wrt->n_elems, std::move(host_array));
}

auto cugrad::tensor::set_grad(const float v) const -> void
{
    auto host_array = std::make_unique<float[]>(n_elems);

    for (int i = 0; i < n_elems; i++)
        host_array[i] = v;

    cudaMemcpy(grad_buffer, host_array.get(), alloc_size, cudaMemcpyHostToDevice);
}

auto cugrad::tensor::alloc() -> void
{
    cudaMalloc(&buffer, alloc_size);
    cudaMalloc(&grad_buffer, alloc_size);
}

auto cugrad::tensor::dealloc() -> void
{
    cudaFree(buffer);
    cudaFree(grad_buffer);
}