#pragma once
#include "Variable.h"
#include "Module.h"

#include <memory>

namespace af
{
    namespace nn
    {

        typedef std::shared_ptr<Module> ModulePtr;

        class Container : public Module
        {
        protected:

            std::vector<ModulePtr> m_modules;

            Container();

        public:

            template<typename T>
            void add(T module)
            {
                m_modules.emplace_back(new T(module));
                for (auto param : module.parameters()) {
                    m_parameters.push_back(param);
                }
            }

            ModulePtr get(int id);

            std::vector<ModulePtr> modules();
        };

        class Sequential : public Container
        {
        public:

            Sequential();

            autograd::Variable forward(const autograd::Variable& input);
        };
    }
}