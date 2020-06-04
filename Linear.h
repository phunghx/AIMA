#pragma once
#include "Module.h"

namespace af
{
    namespace nn
    {
        class Linear : public Module
        {
        private:
            bool m_bias;
        public:
            Linear(int input_size, int output_size, bool bias = true, float spread = 0.05);

            Linear(const autograd::Variable& w);

            Linear(const autograd::Variable& w, const autograd::Variable& b);

            autograd::Variable forward(const autograd::Variable& input);
        };
    }
}