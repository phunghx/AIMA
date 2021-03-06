#include "Variable.h"

#include "Init.h"
#include "Linear.h"

namespace af
{
    namespace nn
    {
        using namespace autograd;

        Linear::Linear(int input_size, int output_size, bool bias, float spread) :
            m_bias(bias)
        {
            auto w = nn::lecunNormal(output_size, input_size);
            if (bias) {
                auto b = nn::lecunNormal(output_size, 1);
                setParams({ w, b });
            }
            else {
                setParams({ w });
            }
        }

        Linear::Linear(const Variable& w) :
            m_bias(false),
            Module({ w })
        {
        }

        Linear::Linear(const Variable& w, const Variable& b) :
            m_bias(true),
            Module({ w, b })
        {
            if (b.array().dims(0) != w.array().dims(0)) {
                throw af::exception("nn:Linear: Dimension mismatch between weight and bias.");
            }
            if (b.array().dims(1) != 1) {
                throw af::exception("nn::Linear: Bias must be a vector.");
            }
        }

        Variable Linear::forward(const Variable& input)
        {
            auto res = matmul(m_parameters[0], input);
            if (m_bias) {
                res = res + tileAs(m_parameters[1], res);
            }
            return res;
        }
    }
}