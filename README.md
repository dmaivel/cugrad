# cugrad ![license](https://img.shields.io/badge/license-MIT-blue)

An automatic differentiation library written in C++ and CUDA. No cuBLAS dependency.

## Getting started

The following libraries are required for building cugrad:
- cuda

The following script will build the library and all the included examples:

```bash
git clone https://github.com/dmaivel/cugrad.git
cd cugrad
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Note

Kernels are mostly unoptimized; do not expect highest performance possible (yet).

## Usage

For use in your own project:
- Link the shared library (`libcugrad.so`)
- Include the primary header (`#include <cugrad.hpp>`)

### Declaring tensors

The library uses shared smart pointers for tensor storage, so it is recommended you use `auto` when creating tensors.

```c++
auto a = cugrad::create_tensor({ 1 }); // create a tensor with shape (1)
auto b = cugrad::create_tensor({ 2, 2 }); // create a tensor with shape (2, 2)
auto c = cugrad::create_tensor({ 3, 4, 6 }); // create a tensor with shape (3, 4, 6)
```

You may use either type in place of `auto`:
```c++
std::shared_ptr<cugrad::tensor> a;
cugrad::tensor_ptr b;
```

You can not use non-pointer tensors.

### Constructing equations

The library provides many operations, all in the namespace `cugrad`.

```c++
auto a = cugrad::create_tensor({ 1 });
auto b = cugrad::create_tensor({ 1 });

auto c = cugrad::add(a, b);
auto d = cugrad::mul(a, c);
```

Note that it is not required to declare each operation as its own seperate variable:
```c++
// equivalent to `f(a, b) = ((a * b) + a) * (b - a)`
auto f(std::shared_ptr<cugrad::tensor> a, std::shared_ptr<cugrad::tensor> b) -> std::shared_ptr<cugrad::tensor>
{
    return cugrad::mul(cugrad::add(cugrad::mul(a, b), a), cugrad::sub(b, a));
}
```

### Setting values

Before computing the results or gradients, you need to set the values found in the input variables. This can be achieved with the following:

```c++
// set every value in `a` to `3.f`
a->set(3.f);
```

### Raw tensors

Before proceeding to the compute sections, its important to take note of another class provided: `cugrad::raw_tensor`. This is simply a storage type for managing buffers returned to the CPU. It is not stored in a smart pointer.

```c++
auto result = /* ... */; // result is raw_tensor

float x = result[0]; // linear access
float y = result({ 1, 2 }) // positional access
```

### Compute result

To compute the result of an equation, simply call `compute()` on the final variable.

```c++
auto y = f(a, b);
y->compute();
```

Because the result is stored on the GPU, you must call `cpu()` to extract the result.

```c++
auto result = y->cpu();
```

You may combine the two calls for the following one-liner:

```c++
auto result = y->compute()->cpu();
```

To extract a specific value, simply index it:

```c++
auto real_result = result[0];
// or
auto real_result = y->compute()->cpu()[0];
// or
auto real_result = y->compute()->cpu()({ 0, 0 }); // assuming 2d shape
```

### Compute gradient

To compute the gradient of an equation, ensure you have first computed a result (using `compute()`). Then, call `grad()`.

```c++
y->grad();
```

To obtain the partial derivative with respect to `a`, we call `cpu(wrt)`.

```c++
auto result = y->cpu(a);
```

Once again, you may combine all these calls into a one-liner:

```c++
// compute() included only if compute() was not previously called
auto result = y->compute()->grad()->cpu(a);
```

As mentioned before, the return type of `cpu(...)` is a raw tensor, so you must index it to extract actual values.