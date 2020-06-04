#include "Variable.h"
#include "Container.h"

namespace af
{
    namespace nn
    {
        using namespace autograd;

        Container::Container() {}

        ModulePtr Container::get(int id)
        {
            return m_modules[id];
        }

        std::vector<ModulePtr> Container::modules()
        {
            return m_modules;
        }

        Sequential::Sequential() {}

        Variable Sequential::forward(const Variable& input)
        {
            Variable output = input;
            for (auto& module : m_modules) {
                output = module->forward(output);
            }
            return output;
        }
    }
}