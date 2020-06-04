#pragma once

#include "Module.h"

namespace af
{
    namespace nn
    {
        class Dropout : public Module
        {
        private:
            double m_ratio;
        public:
            Dropout(double drop_ratio = 0.5);

            autograd::Variable forward(const autograd::Variable& input);
        };
    }
}