#include <cugrad.hpp>
#include <cugrad_kernels.hpp>

#include <functional>
#include <cmath>

constexpr int thr_per_blk = 256;

auto cugrad::shape_to_str(const std::vector<int> &shape) -> std::string
{
    std::string res("(");
    for (const auto & x : shape)
        res += std::to_string(x) + ", ";
    return res + ")";
}

auto cugrad::raw_tensor::operator [](int i) const -> float
{
    CUGRAD_ASSERT(i < n_elems && i > -1, "index (%d) is out of bounds (0, %d)\n", i, n_elems - 1);
    return buffer[i];
}

auto cugrad::raw_tensor::operator [](int i) -> float&
{
    CUGRAD_ASSERT(i < n_elems && i > -1, "index (%d) is out of bounds (0, %d)\n", i, n_elems - 1);
    return buffer[i];
}

auto cugrad::raw_tensor::operator ()(const std::vector<int> &pos) const -> float
{
    CUGRAD_ASSERT(pos.size() == dims.size(), "position size (%ld) does not match dimensions (%ld)\n", pos.size(), dims.size());
    for (int i = 0; i < pos.size(); i++)
        CUGRAD_ASSERT(pos[i] < dims[i] && pos[i] > -1, "index (%d) at pos[%d] is out of bounds (0, %d)\n", pos[i], i, dims[i] - 1);

    int index = 0;
    int stride = 1;

    for (int i = dims.size() - 1; i >= 0; i--) {
        index += pos[i] * stride;
        stride *= dims[i];
    }

    return buffer[index];
}

auto cugrad::tensor::shape() const -> std::vector<int>
{
    return dims;
}

auto cugrad::tensor::compute() -> std::shared_ptr<cugrad::tensor>
{
    std::function<void(std::shared_ptr<cugrad::tensor> var)> compute;

    compute = [&](std::shared_ptr<cugrad::tensor> var) -> void {
        for (auto src : var->src_tensors) {
            compute(src);

            if (src->op == CUGRAD_OP_NONE)
                continue;

            src->compute();
        }
    };

    compute(get_shared_ptr());

    int blk_in_grid = std::ceil(float(n_elems) / thr_per_blk);

    switch (op) {
    case CUGRAD_OP_ADD:
        cugrad::kernel::add(blk_in_grid, thr_per_blk, n_elems, static_cast<float*>(buffer), static_cast<float*>(src_tensors[0]->buffer), static_cast<float*>(src_tensors[1]->buffer));
        break;
    case CUGRAD_OP_SUB:
        cugrad::kernel::sub(blk_in_grid, thr_per_blk, n_elems, static_cast<float*>(buffer), static_cast<float*>(src_tensors[0]->buffer), static_cast<float*>(src_tensors[1]->buffer));
        break;
    case CUGRAD_OP_MUL:
        cugrad::kernel::mul(blk_in_grid, thr_per_blk, n_elems, static_cast<float*>(buffer), static_cast<float*>(src_tensors[0]->buffer), static_cast<float*>(src_tensors[1]->buffer));
        break;
    case CUGRAD_OP_DIV:
        cugrad::kernel::div(blk_in_grid, thr_per_blk, n_elems, static_cast<float*>(buffer), static_cast<float*>(src_tensors[0]->buffer), static_cast<float*>(src_tensors[1]->buffer));
        break;
    case CUGRAD_OP_MATMUL:
        cugrad::kernel::matmul( blk_in_grid, thr_per_blk, n_elems, src_tensors[1]->shape()[0], static_cast<float*>(buffer), static_cast<float*>(src_tensors[0]->buffer), static_cast<float*>(src_tensors[1]->buffer));
        break;
    case CUGRAD_OP_SUM:
        blk_in_grid = std::ceil(float(src_tensors[0]->n_elems) / thr_per_blk);
        cugrad::kernel::sum(blk_in_grid, thr_per_blk, src_tensors[0]->n_elems, static_cast<float*>(buffer), static_cast<float*>(src_tensors[0]->buffer));
        break;
    case CUGRAD_OP_EXP:
        cugrad::kernel::exp(blk_in_grid, thr_per_blk, n_elems, static_cast<float*>(buffer), static_cast<float*>(src_tensors[0]->buffer));
        break;
    case CUGRAD_OP_LOG:
        cugrad::kernel::log(blk_in_grid, thr_per_blk, n_elems, static_cast<float*>(buffer), static_cast<float*>(src_tensors[0]->buffer));
        break;
    case CUGRAD_OP_SQRT:
        cugrad::kernel::sqrt(blk_in_grid, thr_per_blk, n_elems, static_cast<float*>(buffer), static_cast<float*>(src_tensors[0]->buffer));
        break;
    default:
        CUGRAD_ASSERT(false, "opcode (%d) is unimplemented\n", op);
    }

    return get_shared_ptr();
}

auto cugrad::tensor::grad() -> std::shared_ptr<cugrad::tensor>
{
    std::function<void(std::shared_ptr<cugrad::tensor> var)> compute, reset_grads;

    reset_grads = [&](std::shared_ptr<cugrad::tensor> var) -> void {
        for (auto src : var->src_tensors) {
            src->set_grad(0.f);
            reset_grads(src);
        }
    };

    compute = [&](std::shared_ptr<cugrad::tensor> var) -> void {
        for (auto src : var->src_tensors) {                
            if (var->op == CUGRAD_OP_NONE)
                continue;

            int blk_in_grid = ceil( float(var->n_elems) / thr_per_blk );

            switch (var->op) {
            case CUGRAD_OP_ADD:
                cugrad::kernel::add_grad(blk_in_grid, thr_per_blk, var->n_elems, static_cast<float*>(src->grad_buffer), static_cast<float*>(var->grad_buffer));
                break;
            case CUGRAD_OP_SUB:
                cugrad::kernel::sub_grad(blk_in_grid, thr_per_blk, var->n_elems, static_cast<float*>(src->grad_buffer), static_cast<float*>(var->grad_buffer), var->src_tensors[1] == src);
                break;
            case CUGRAD_OP_MUL:
                cugrad::kernel::mul_grad( blk_in_grid, thr_per_blk, var->n_elems
                                        , static_cast<float*>(src->grad_buffer)
                                        , static_cast<float*>(var->grad_buffer)
                                        , var->src_tensors[0] == src ? static_cast<float*>(var->src_tensors[1]->buffer) : static_cast<float*>(var->src_tensors[0]->buffer) );
                break;
            case CUGRAD_OP_DIV:
                cugrad::kernel::div_grad( blk_in_grid, thr_per_blk, var->n_elems
                                        , static_cast<float*>(src->grad_buffer)
                                        , static_cast<float*>(var->grad_buffer)
                                        , static_cast<float*>(var->src_tensors[0]->buffer)
                                        , static_cast<float*>(var->src_tensors[1]->buffer)
                                        , var->src_tensors[1] == src );
                break;
            case CUGRAD_OP_MATMUL:
                cugrad::kernel::matmul_grad( blk_in_grid, thr_per_blk, var->n_elems, var->src_tensors[1]->shape()[0]
                                           , static_cast<float*>(src->grad_buffer)
                                           , static_cast<float*>(var->grad_buffer)
                                           , static_cast<float*>(var->src_tensors[0]->buffer)
                                           , static_cast<float*>(var->src_tensors[1]->buffer)
                                           , var->src_tensors[0] == src );
                break;
            case CUGRAD_OP_SUM:
                blk_in_grid = std::ceil(float(src->n_elems) / thr_per_blk);
                cugrad::kernel::sum_grad( blk_in_grid, thr_per_blk, src->n_elems
                                        , static_cast<float*>(src->grad_buffer)
                                        , static_cast<float*>(var->grad_buffer) );
                break;
            case CUGRAD_OP_EXP:
                cugrad::kernel::exp_grad( blk_in_grid, thr_per_blk, var->n_elems
                                        , static_cast<float*>(src->grad_buffer)
                                        , static_cast<float*>(var->grad_buffer)
                                        , static_cast<float*>(src->buffer) );
                break;
            case CUGRAD_OP_LOG:
                cugrad::kernel::log_grad( blk_in_grid, thr_per_blk, var->n_elems
                                        , static_cast<float*>(src->grad_buffer)
                                        , static_cast<float*>(var->grad_buffer)
                                        , static_cast<float*>(src->buffer) );
                break;
            case CUGRAD_OP_SQRT:
                cugrad::kernel::sqrt_grad( blk_in_grid, thr_per_blk, var->n_elems
                                        , static_cast<float*>(src->grad_buffer)
                                        , static_cast<float*>(var->grad_buffer)
                                        , static_cast<float*>(src->buffer) );
                break;
            default:
                CUGRAD_ASSERT(false, "opcode (%d) is unimplemented\n", var->op);
            }

            compute(src);
        }
    };

    auto this_ptr = get_shared_ptr();

    set_grad(1.f);
    reset_grads(this_ptr);
    compute(this_ptr);

    return this_ptr;
}

auto cugrad::tensor::get_shared_ptr() -> std::shared_ptr<cugrad::tensor>
{
    return shared_from_this();
}

auto cugrad::add(std::shared_ptr<cugrad::tensor> src0, std::shared_ptr<cugrad::tensor> src1) -> std::shared_ptr<cugrad::tensor>
{
    CUGRAD_ASSERT(src0->shape() == src1->shape(), "src0 has shape %s while src1 has shape %s\n", shape_to_str(src0->shape()).c_str(), shape_to_str(src1->shape()).c_str());
    return std::make_shared<cugrad::tensor>(src0->shape(), CUGRAD_OP_ADD, std::vector<std::shared_ptr<cugrad::tensor>>{ src0, src1 });
}

auto cugrad::sub(std::shared_ptr<cugrad::tensor> src0, std::shared_ptr<cugrad::tensor> src1) -> std::shared_ptr<cugrad::tensor>
{
    CUGRAD_ASSERT(src0->shape() == src1->shape(), "src0 has shape %s while src1 has shape %s\n", shape_to_str(src0->shape()).c_str(), shape_to_str(src1->shape()).c_str());
    return std::make_shared<cugrad::tensor>(src0->shape(), CUGRAD_OP_SUB, std::vector<std::shared_ptr<cugrad::tensor>>{ src0, src1 });
}

auto cugrad::mul(std::shared_ptr<cugrad::tensor> src0, std::shared_ptr<cugrad::tensor> src1) -> std::shared_ptr<cugrad::tensor>
{
    CUGRAD_ASSERT(src0->shape() == src1->shape(), "src0 has shape %s while src1 has shape %s\n", shape_to_str(src0->shape()).c_str(), shape_to_str(src1->shape()).c_str());
    return std::make_shared<cugrad::tensor>(src0->shape(), CUGRAD_OP_MUL, std::vector<std::shared_ptr<cugrad::tensor>>{ src0, src1 });
}

auto cugrad::div(std::shared_ptr<cugrad::tensor> src0, std::shared_ptr<cugrad::tensor> src1) -> std::shared_ptr<cugrad::tensor>
{
    CUGRAD_ASSERT(src0->shape() == src1->shape(), "src0 has shape %s while src1 has shape %s\n", shape_to_str(src0->shape()).c_str(), shape_to_str(src1->shape()).c_str());
    return std::make_shared<cugrad::tensor>(src0->shape(), CUGRAD_OP_DIV, std::vector<std::shared_ptr<cugrad::tensor>>{ src0, src1 });
}

auto cugrad::matmul(std::shared_ptr<cugrad::tensor> src0, std::shared_ptr<cugrad::tensor> src1) -> std::shared_ptr<cugrad::tensor>
{
    bool same_size = src0->shape() == src1->shape();
    bool shapes_2x1 = (src0->shape().size() == src1->shape().size() + 1) || (src0->shape().size() + 1 == src1->shape().size());

    CUGRAD_ASSERT(same_size || shapes_2x1, "src0 has shape %s while src1 has shape %s\n", shape_to_str(src0->shape()).c_str(), shape_to_str(src1->shape()).c_str());

    bool switch_order = src0->shape().size() > src1->shape().size();

    auto new_shape = src1->shape();
    new_shape.erase(new_shape.begin());

    return std::make_shared<cugrad::tensor>(new_shape, CUGRAD_OP_MATMUL, 
        !switch_order ? std::vector<std::shared_ptr<cugrad::tensor>>{ src0, src1 } : std::vector<std::shared_ptr<cugrad::tensor>>{ src1, src0 });
}

auto cugrad::sum(std::shared_ptr<cugrad::tensor> src) -> std::shared_ptr<cugrad::tensor>
{
    return std::make_shared<cugrad::tensor>(std::vector<int>{ 1 }, CUGRAD_OP_SUM, std::vector<std::shared_ptr<cugrad::tensor>>{ src });
}

auto cugrad::exp(std::shared_ptr<cugrad::tensor> src) -> std::shared_ptr<cugrad::tensor>
{
    return std::make_shared<cugrad::tensor>(src->shape(), CUGRAD_OP_EXP, std::vector<std::shared_ptr<cugrad::tensor>>{ src });
}

auto cugrad::log(std::shared_ptr<cugrad::tensor> src) -> std::shared_ptr<cugrad::tensor>
{
    return std::make_shared<cugrad::tensor>(src->shape(), CUGRAD_OP_LOG, std::vector<std::shared_ptr<cugrad::tensor>>{ src });
}

auto cugrad::sqrt(std::shared_ptr<cugrad::tensor> src) -> std::shared_ptr<cugrad::tensor>
{
    return std::make_shared<cugrad::tensor>(src->shape(), CUGRAD_OP_SQRT, std::vector<std::shared_ptr<cugrad::tensor>>{ src });
}

auto cugrad::create_tensor(const std::vector<int> &shape) -> std::shared_ptr<cugrad::tensor>
{
    return std::make_shared<cugrad::tensor>(shape);
}