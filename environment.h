#include <arrayfire.h>
#include <iostream>
#include <list>
#include "agent.h"

class Environment {
public:
	std::list<Thing> things;
	std::list<Agent> agents;
	Environment();
	af::array  percept(Agent&);
	virtual af::array execute_action(Agent&, int);
	af::array default_location(Thing thing);
	bool is_done();
	virtual void step();	
};