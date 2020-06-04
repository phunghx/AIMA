#include <iostream>
#include <arrayfire.h>
#include "agent.h"


#include "autograd.h"
#include "nn.h"
#include "optim.h"

#include <string>
#include <memory>

using namespace af;
using namespace af::nn;
using namespace af::autograd;

int main(int argc, const char** args) {
    /*
	std::vector<std::string> actions;
	actions.push_back("left");
	actions.push_back("right");
	actions.push_back("suck");
	actions.push_back("NoOp");

	Program* program = RandomAgentProgram(actions);
	Agent agent(program);
	*/

    int optim_mode = 0;
    std::string optimizer_arg = "--adam";
    /*std::string optimizer_arg = std::string(args[1]);
    if (optimizer_arg == "--adam") {
        optim_mode = 1;
    }
    else if (optimizer_arg == "--rmsprop") {
        optim_mode = 2;
    }
    else {

    }*/
    optim_mode = 1;

    const int inputSize = 2;
    const int outputSize = 1;
    const double lr = 0.01;
    const double mu = 0.1;
    const int numSamples = 4;

    float hInput[] = { 1, 1,
                      0, 0,
                      1, 0,
                      0, 1 };

    float hOutput[] = { 1,
                       0,
                       1,
                       1 };

    auto in = af::array(inputSize, numSamples, hInput);
    auto out = af::array(outputSize, numSamples, hOutput);

    nn::Sequential model;

    model.add(nn::Linear(inputSize, outputSize));
    model.add(nn::Sigmoid());

    auto loss = nn::MeanSquaredError();

    std::unique_ptr<optim::Optimizer> optim;

    if (optimizer_arg == "--rmsprop") {
        optim = std::unique_ptr<optim::Optimizer>(new optim::RMSPropOptimizer(model.parameters(), lr));
    }
    else if (optimizer_arg == "--adam") {
        optim = std::unique_ptr<optim::Optimizer>(new optim::AdamOptimizer(model.parameters(), lr));
    }
    else {
        optim = std::unique_ptr<optim::Optimizer>(new optim::SGDOptimizer(model.parameters(), lr, mu));
    }

    Variable result, l;
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < numSamples; j++) {

            model.train();
            optim->zeroGrad();

            af::array in_j = in(af::span, j);
            af::array out_j = out(af::span, j);

            // Forward propagation
            result = model(nn::input(in_j));

            // Calculate loss
            l = loss(result, nn::noGrad(out_j));

            // Backward propagation
            l.backward();

            // Update parameters
            optim->update();
        }

        if ((i + 1) % 100 == 0) {
            model.eval();

            // Forward propagation
            result = model(nn::input(in));

            // Calculate loss
            // TODO: Use loss function
            af::array diff = out - result.array();
            printf("Average Error at iteration(%d) : %lf\n", i + 1, af::mean<float>(af::abs(diff)));
            printf("Predicted\n");
            af_print(result.array());
            printf("Expected\n");
            af_print(out);
            printf("\n\n");
        }
    }

	return 1;
}