#pragma once
#include <arrayfire.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>
#undef max
#undef min
namespace af {
    namespace autograd {
        class Variable
        {
        public:
            typedef std::function<void(std::vector<Variable>&, const Variable&)> GradFunc_t;
            typedef std::unordered_map<std::ptrdiff_t, bool> Cache_t;
            typedef std::vector<Variable> DAG_t;

        private:
            struct Shared {
                Shared();
                Shared(const af::array& data, bool calc_grad);
                Shared(const af::array& data,
                    const std::vector<Variable>& inputs,
                    GradFunc_t grad_func,
                    bool calc_grad);

                bool m_calc_grad;
                af::array m_data;
                std::vector<Variable> m_inputs;
                std::vector<Variable> m_grads;
                GradFunc_t m_grad_func;
            };

        public:

            Variable();
            Variable(const af::array& data, bool calc_grad);
            Variable(const af::array& data,
                const std::vector<Variable>& inputs,
                GradFunc_t grad_func);

            af::array& array() const;

            Variable& grad() const;

            std::ptrdiff_t id() const;

            bool isCalcGrad() const;

            bool isGradAvailable() const;

            af::dim4 dims() const;

            af::dtype type() const;

            void zeroGrad();

            void setCalcGrad(bool calc_grad);

            void addGrad(const Variable& child_grad);

            void calcGradInputs(bool retain_grad_graph = false);

            void backward(const Variable& grad, bool retain_grad_graph = false);

            void backward(bool retain_grad_graph = false);

        private:
            void evalGrad(bool retain_grad_graph = false);

            std::vector<Variable>& getInputs() const;

            static void buildSubGraph(Cache_t& cache, DAG_t& dag, const Variable& var);

            static DAG_t build(const Variable& var);

            std::shared_ptr<Shared> m_shared;
            friend  Variable operator +(const Variable& lhs, const Variable& rhs);
            friend Variable operator *(const Variable& lhs, const Variable& rhs);
            friend Variable operator -(const Variable& lhs, const Variable& rhs);
            friend Variable operator /(const Variable& lhs, const Variable& rhs);
            friend Variable operator >(const Variable& lhs, const Variable& rhs);
            friend Variable operator <(const Variable& lhs, const Variable& rhs);
            friend Variable operator >=(const Variable& lhs, const Variable& rhs);
            friend Variable operator <=(const Variable& lhs, const Variable& rhs);

            friend Variable operator +(const double& lhs, const Variable& rhs);
            friend  Variable operator *(const double& lhs, const Variable& rhs);
            friend Variable operator -(const double& lhs, const Variable& rhs);
            friend Variable operator /(const double& lhs, const Variable& rhs);
            friend Variable operator >(const double& lhs, const Variable& rhs);
            friend Variable operator <(const double& lhs, const Variable& rhs);
            friend Variable operator >=(const double& lhs, const Variable& rhs);
            friend Variable operator <=(const double& lhs, const Variable& rhs);

            friend  Variable operator +(const Variable& lhs, const double& rhs);
            friend Variable operator *(const Variable& lhs, const double& rhs);
            friend  Variable operator -(const Variable& lhs, const double& rhs);
            friend  Variable operator /(const Variable& lhs, const double& rhs);
            friend Variable operator >(const Variable& lhs, const double& rhs);
            friend Variable operator <(const Variable& lhs, const double& rhs);
            friend Variable operator >=(const Variable& lhs, const double& rhs);
            friend Variable operator <=(const Variable& lhs, const double& rhs);

            friend  Variable operator !(const Variable& input);

            friend  Variable negate(const Variable& input);
            friend  Variable reciprocal(const Variable& input);

            friend  Variable exp(const Variable& input);
            friend  Variable log(const Variable& input);
            friend  Variable sin(const Variable& input);
            friend  Variable cos(const Variable& input);
            friend  Variable tanh(const Variable& input);
            friend  Variable sigmoid(const Variable& input);

            friend  Variable max(const Variable& lhs, const Variable& rhs);
            friend  Variable max(const Variable& lhs, const double& rhs);
            friend  Variable max(const double& lhs, const Variable& rhs);

            friend  Variable min(const Variable& lhs, const Variable& rhs);
            friend  Variable min(const Variable& lhs, const double& rhs);
            friend  Variable min(const double& lhs, const Variable& rhs);

            friend  Variable transpose(const Variable& input);
            friend  Variable tileAs(const Variable& input, const Variable& reference);
            friend  Variable sumAs(const Variable& input, const Variable& reference);

            friend  Variable tile(const Variable& input, const std::vector<int>& repeats);
            friend  Variable sum(const Variable& input, const std::vector<int>& axes);
            friend  Variable mean(const Variable& input, const std::vector<int>& axes);

            friend  Variable matmul(const Variable& lhs, const Variable& rhs);
            friend  Variable matmulTN(const Variable& lhs, const Variable& rhs);
            friend  Variable matmulNT(const Variable& lhs, const Variable& rhs);

            friend   Variable abs(const Variable& input);

            friend  Variable flat(const Variable& input);
            friend  Variable moddims(const Variable& input, const af::dim4& dims);

        };
    }
}