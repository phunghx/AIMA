#include "Variable.h"
#include "Loss.h"


namespace af
{
    namespace nn
    {
        using namespace autograd;

        autograd::Variable Loss::forward(const autograd::Variable& inputs)
        {
            throw af::exception("Loss module requires both inputs and targets");
        }

        autograd::Variable Loss::operator()(const autograd::Variable& inputs,
            const autograd::Variable& targets)
        {
            return this->forward(inputs, targets);
        }

        autograd::Variable MeanSquaredError::forward(const autograd::Variable& inputs,
            const autograd::Variable& targets)
        {
            auto df = inputs - targets;
            auto res = mean(flat(df * df), { 0 });
            return res;
        }

        autograd::Variable MeanAbsoluteError::forward(const autograd::Variable& inputs,
            const autograd::Variable& targets)
        {
            auto df = inputs - targets;
            auto res = mean(flat(abs(df)), { 0 });
            return res;
        }

        static autograd::Variable
            binaryCrossEntropy(const autograd::Variable& inputs,
                const autograd::Variable& targets)
        {
            return targets* inputs + (1 - targets) * (1 - inputs);
        }

        autograd::Variable BinaryCrossEntropyLoss::forward(const autograd::Variable& inputs,
            const autograd::Variable& targets)
        {
            return mean(flat(binaryCrossEntropy(inputs, targets)), { 0 });
        }

        autograd::Variable BinaryCrossEntropyLoss::forward(const autograd::Variable& inputs,
            const autograd::Variable& targets,
            const autograd::Variable& weights)
        {
            return mean(flat(weights * binaryCrossEntropy(inputs, targets)), { 0 });
        }
    }
}