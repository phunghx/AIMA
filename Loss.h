#pragma once

#include "Module.h"

namespace af
{
    namespace nn
    {
        class Loss : public Module
        {
        public:
            Loss() {}

            virtual autograd::Variable forward(const autograd::Variable& inputs,
                const autograd::Variable& targets) = 0;

            autograd::Variable forward(const autograd::Variable& inputs);

            autograd::Variable operator()(const autograd::Variable& inputs,
                const autograd::Variable& targets);
        };

        class MeanSquaredError : public Loss
        {
        public:
            MeanSquaredError() {}

            autograd::Variable forward(const autograd::Variable& inputs,
                const autograd::Variable& targets);
        };

        class MeanAbsoluteError : public Loss
        {
        public:
            MeanAbsoluteError() {}

            autograd::Variable forward(const autograd::Variable& inputs,
                const autograd::Variable& targets);
        };

        class BinaryCrossEntropyLoss : public Loss
        {
        public:
            BinaryCrossEntropyLoss() {}

            autograd::Variable forward(const autograd::Variable& inputs,
                const autograd::Variable& targets);

            autograd::Variable forward(const autograd::Variable& inputs,
                const autograd::Variable& targets,
                const autograd::Variable& weights);
        };

        typedef MeanSquaredError MSE;
        typedef MeanAbsoluteError MAE;
        typedef MeanAbsoluteError L1Loss;
        typedef BinaryCrossEntropyLoss BCELoss;
    }
}