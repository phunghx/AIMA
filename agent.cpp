#include "agent.h"

Thing::operator std::string() const
{
	return this->name;
}

bool Thing::is_alive(void)
{
	return this->alive;
}

void Thing::show_state(void)
{
	std::cout << "state\n";
}

void Thing::display(int x, int y, int width, int height)
{
	std::cout << "display\n";
}

Program::Program()
{
}

std::string Program::run(af::array percept)
{
	return "";
}

Program* Program::copy(void)
{
	return  new Program(*this);
}



Agent::Agent(Program* _program)
{
	this->alive = true;
	this->bump = false;
	this->holding.clear();
	this->program = _program;
	this->performance = 0;
}

Agent TraceAgent(Agent agent) {
	Program* old_program = agent.program->copy();
	class new_program: public Program {
	public:
		Program* old_program;
		new_program(Program* old_program) {
			this->old_program = old_program;
		};
		std::string run(af::array percept) override {
			std::string action = old_program->run(percept);
			std::cout << "\n";
			return action;
		};	
	};
	agent.program = new new_program(old_program);
	return agent;
}
Program* RandomAgentProgram(std::vector<std::string> actions)
{
	class new_program : public Program {
	public:
		std::vector<std::string> actions;
		new_program(std::vector<std::string>& actions) {
			this->actions = actions;
		}
		std::string run(af::array percept) override {
			return std::to_string(std::rand() % this->actions.size());
		}
	};
	return new new_program(actions);
}
Program* TableDrivenAgentProgram(std::map<std::string, std::string> table, Model model) {
	std::list<af::array> percepts;
	class new_program : public Program {
	public:
		std::list<af::array> percepts;
		std::map<std::string, std::string> table;
		Model model;
		new_program(std::list<af::array> &percepts, std::map<std::string, std::string> &table,
			Model & model) {
			this->table = table;
			this->percepts = percepts;
			this->model = model;
		};
		std::string predict(std::list<af::array>& percepts) {
			return this->model.predict(percepts);
		}
		std::string run(af::array percept) override {
			this->percepts.push_back(percept);
			std::cout << "\n";
			return this->table[this->predict(percepts)];
		};
	};
	return new new_program(percepts,table, model);

}

Model::Model()
{
}

std::vector<int> Model::encode(std::list<af::array>)
{
	return std::vector<int>();
}

std::string Model::predict(std::list<af::array> percepts)
{
	return model[this->encode(percepts)];
}
