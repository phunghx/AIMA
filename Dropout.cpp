#include "Variable.h"

#include "Init.h"
#include "Dropout.h"

namespace af
{
    namespace nn
    {
        using namespace autograd;

        Dropout::Dropout(double drop_ratio) :
            m_ratio(drop_ratio)
        {
        }

        Variable Dropout::forward(const Variable& input)
        {
            if (m_train)
                return (uniform(input.dims(), 0.0, 1.0, f32, false) > m_ratio) * input;
            else
                return input;
        }
    }
}