#pragma once
#include <string>
#include <arrayfire.h>
#include <iostream>
#include <list>
#include <map>
#include <vector>

using namespace af;
class Thing {
public:
	std::string name;
	bool alive;
public:
	operator std::string() const;
	bool is_alive(void);
	void show_state(void);
	void display(int, int, int, int);
};
class Program {
public:
	
	Program();
	
	virtual std::string run (af::array);
	virtual Program* copy(void);
};
class Agent : public Thing {
public:
	bool bump;
	float performance;
	std::list<int> holding;
	Program* program;
public:
	Agent(Program* _program);
};

class Model {
public:
	std::map<std::vector<int>, std::string> model;
	Model();
	std::vector<int> encode(std::list<af::array>);
	std::string predict(std::list<af::array>);
};

Agent TraceAgent(Agent);
Program* TableDrivenAgentProgram(std::map<std::string,int>);
Program* RandomAgentProgram(std::vector<std::string> actions);
