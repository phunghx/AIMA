#include "environment.h"

Environment::Environment()
{
	this->things.clear();
	this->agents.clear();
}

af::array Environment::percept(Agent& agent)
{
	return af::array(1);
}

af::array Environment::execute_action(Agent& agent, int actioin)
{
	return af::array();
}

af::array Environment::default_location(Thing thing)
{
	return NULL;
}

bool Environment::is_done()
{
	for (std::list<Agent>::iterator it = this->agents.begin(); it != this->agents.end(); it++) {
		if (it->alive == true)
			return false;
	}
	return true;
}

void Environment::step()
{
	if (!this->is_done()) {
		std::list<std::string> actions;
		for (std::list<Agent>::iterator it = this->agents.begin(); it != this->agents.end(); it++) {
			if (it->alive) {
				actions.push_back(it->program->run(this->percept(*it)));
			}
			else{
				actions.push_back("");
			}
		}
	}
}
