#pragma once
#include "Variable.h"
#include <string>
#include <vector>

namespace af
{
    namespace nn
    {

        class Module
        {
        protected:
            std::vector<autograd::Variable> m_parameters;

            bool m_train;

            Module();

            Module(const std::vector<autograd::Variable>& parameters);

            void setParams(const std::vector<autograd::Variable>& parameters);

        public:

            std::vector<autograd::Variable> parameters();

            void train();

            void eval();

            virtual autograd::Variable forward(const autograd::Variable& input) = 0;

            autograd::Variable operator()(const autograd::Variable& input);
        };
    }
}