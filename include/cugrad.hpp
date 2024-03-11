#pragma once

#include <vector>
#include <memory>
#include <cstdio>
#include <string>

#define CUGRAD_ASSERT(x, ...) \
    do { \
        if (!(x)) { \
            fflush(stdout); \
            fprintf(stderr, "CUGRAD_ASSERT: %s:%d:%s: %s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, #x); \
            fprintf(stderr, " || " __VA_ARGS__); \
            abort(); \
        } \
    } while (0)

enum operation {
    CUGRAD_OP_NONE,
    CUGRAD_OP_ADD,
    CUGRAD_OP_SUB,
    CUGRAD_OP_MUL,
    CUGRAD_OP_DIV,
    CUGRAD_OP_MATMUL,
    CUGRAD_OP_SUM,
    CUGRAD_OP_EXP,
    CUGRAD_OP_LOG,
    CUGRAD_OP_SQRT
};

enum optimizer {
    CUGRAD_OPT_NONE,
    CUGRAD_OPT_ADAM
};

namespace cugrad {
    auto shape_to_str(const std::vector<int> &shape) -> std::string;

    class raw_tensor {
    public:
        raw_tensor(const std::vector<int> &shape, int elems, std::unique_ptr<float[]> buf = std::unique_ptr<float[]>(nullptr)) : dims(shape), n_elems(elems), buffer(std::move(buf)) 
        { 
            if (buffer.get() == nullptr)
                buffer = std::make_unique<float[]>(n_elems);
        }

        // in case element count wasnt cached
        raw_tensor(const std::vector<int> &shape, std::unique_ptr<float[]> buf = std::unique_ptr<float[]>(nullptr)) : dims(shape), buffer(std::move(buf)) 
        { 
            n_elems = 1;
            for (const auto & x : dims)
                n_elems *= x;

            if (buffer.get() == nullptr)
                buffer = std::make_unique<float[]>(n_elems);
        }

        auto operator [](int i) const -> float;
        auto operator [](int i) -> float&;
        auto operator ()(const std::vector<int> &pos) const -> float;

        auto shape() const -> std::vector<int>
        {
            return dims;
        }
    private:
        std::unique_ptr<float[]> buffer;

        std::vector<int> dims;
        int n_elems = 0;
    };

    class tensor : public std::enable_shared_from_this<tensor> {
    public:
        tensor(const std::vector<int> &shape, operation o = CUGRAD_OP_NONE, const std::vector<std::shared_ptr<tensor>> &src = {}) : dims(shape), op(o), src_tensors(src)
        {
            n_elems = 1;
            for (const auto & x : dims)
                n_elems *= x;

            alloc_size = n_elems * sizeof(float);

            alloc();
        }

        ~tensor()
        {
            dealloc();
        }

        auto shape() const -> std::vector<int>;

        auto set(const float v) const -> void;
        auto set(const cugrad::raw_tensor &tensor) const -> void;
        auto set_rand(const float min = -1.f, const float max = 1.f) const -> void;

        auto compute() -> std::shared_ptr<tensor>;
        auto grad() -> std::shared_ptr<tensor>;

        auto cpu() const -> cugrad::raw_tensor;
        auto cpu(std::shared_ptr<tensor> wrt) const -> cugrad::raw_tensor;

    private:
        void *buffer = nullptr;
        void *grad_buffer = nullptr;

        std::size_t alloc_size = 0;
        int n_elems = 0;

        std::vector<int> dims;
        std::vector<std::shared_ptr<tensor>> src_tensors;

        operation op;

        auto alloc() -> void;
        auto dealloc() -> void;

        auto set_grad(const float v) const -> void;
        auto get_shared_ptr() -> std::shared_ptr<tensor>;
    };

    using tensor_ptr = std::shared_ptr<tensor>;

    auto create_tensor(const std::vector<int> &shape) -> std::shared_ptr<tensor>;

    auto add(std::shared_ptr<tensor> src0, std::shared_ptr<tensor> src1) -> std::shared_ptr<tensor>;
    auto sub(std::shared_ptr<tensor> src0, std::shared_ptr<tensor> src1) -> std::shared_ptr<tensor>;
    auto mul(std::shared_ptr<tensor> src0, std::shared_ptr<tensor> src1) -> std::shared_ptr<tensor>;
    auto div(std::shared_ptr<tensor> src0, std::shared_ptr<tensor> src1) -> std::shared_ptr<tensor>;

    auto matmul(std::shared_ptr<tensor> src0, std::shared_ptr<tensor> src1) -> std::shared_ptr<tensor>;

    auto sum(std::shared_ptr<tensor> src) -> std::shared_ptr<tensor>;
    auto exp(std::shared_ptr<tensor> src) -> std::shared_ptr<tensor>;
    auto log(std::shared_ptr<tensor> src) -> std::shared_ptr<tensor>;
    auto sqrt(std::shared_ptr<tensor> src) -> std::shared_ptr<tensor>;
}