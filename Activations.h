#pragma once
#include "Variable.h"
#include "Module.h"

namespace af
{
    namespace nn
    {
        class Sigmoid : public Module
        {
        public:
            Sigmoid();

            autograd::Variable forward(const autograd::Variable& input);
        };

        class Tanh : public Module
        {
        public:
            Tanh();

            autograd::Variable forward(const autograd::Variable& input);
        };

        class ReLU : public Module
        {
        public:
            ReLU();

            autograd::Variable forward(const autograd::Variable& input);
        };

        class LeakyReLU : public Module
        {
        private:
            double m_slope;
        public:
            LeakyReLU(double slope = 0.0);

            autograd::Variable forward(const autograd::Variable& input);
        };

        class PReLU : public Module
        {
        public:
            PReLU(int size, double value = 1.0);
            PReLU(const autograd::Variable& w);

            autograd::Variable forward(const autograd::Variable& input);
        };

        class ELU : public Module
        {
        private:
            double m_alpha;
        public:
            ELU(double alpha = 1.0);

            autograd::Variable forward(const autograd::Variable& input);
        };

        class ThresholdReLU : public Module
        {
        private:
            double m_threshold;
        public:
            ThresholdReLU(double threshold = 1.0);

            autograd::Variable forward(const autograd::Variable& input);
        };



    }
}