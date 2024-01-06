/*
 * Copyright 2003-2005 Carnegie Mellon University and Rutgers University
 * Copyright 2007 Håkan Younes
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <config.h>
#include "client.h"
#include "states.h"
#include "problems.h"
#include "domains.h"
#include "actions.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstdio>
#include <cstring>
#if HAVE_GETOPT_LONG
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <getopt.h>
#else
#include "port/getopt.h"
#endif
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>


/* The parse function. */
extern int yyparse();
/* File to parse. */
extern FILE* yyin;
/* Name of current file. */
std::string current_file;
/* Level of warnings. */
int warning_level;
/* Verbosity level. */
int verbosity;

/* Program options. */
static struct option long_options[] = {
  { "actions", optional_argument, 0, 'a' },
  { "host", required_argument, 0, 'H' },
  { "port", required_argument, 0, 'P' },
  { "verbose", optional_argument, 0, 'v' },
  { "warnings", optional_argument, 0, 'W' },
  { "help", no_argument, 0, 'h' },
  { 0, 0, 0, 0 }
};
static const char OPTION_STRING[] = "a:H:P:v::W::h";


/* Displays help. */
static void display_help() {
  std::cout << "usage: mdpclient [options] [file ...]" << std::endl
            << "options:" << std::endl
            << "  -a f,  --actions=f\t"
            << "use action file f" << std::endl
            << "  -H h,  --host=h\t"
            << "connect to host h" << std::endl
            << "  -P p,  --port=p\t"
            << "connect to port p" << std::endl
            << "  -v[n], --verbose[=n]\t"
            << "use verbosity level n;" << std::endl
            << "\t\t\t  n is a number from 0 (verbose mode off) and up;"
            << std::endl
            << "\t\t\t  default level is 1 if optional argument is left out"
            << std::endl
            << "  -W[n], --warnings[=n]\t"
            << "determines how warnings are treated;" << std::endl
            << "\t\t\t  0 supresses warnings; 1 displays warnings;"
            << std::endl
            << "\t\t\t  2 treats warnings as errors" << std::endl
            << "  -h     --help\t\t"
            << "display this help and exit" << std::endl
            << "  file ...\t\t"
            << "files containing domain and problem descriptions;" << std::endl
            << "\t\t\t  if none, descriptions are read from standard input"
            << std::endl
            << std::endl
            << "Report bugs to <" PACKAGE_BUGREPORT ">." << std::endl;
}


/* Parses the given file, and returns true on success. */
static bool read_file(const char* name) {
  yyin = fopen(name, "r");
  if (yyin == 0) {
    std::cerr << "mdpclient:" << name << ": " << strerror(errno)
              << std::endl;
    return false;
  } else {
    current_file = name;
    bool success = (yyparse() == 0);
    fclose(yyin);
    return success;
  }
}

int connect(const char *hostname, int port)
{
  struct hostent *host = ::gethostbyname(hostname);
  if (!host) {
    perror("gethostbyname");
    return -1;
  }

  int sock = ::socket(PF_INET, SOCK_STREAM, 0);
  if (sock == -1) {
    perror("socket");
    return -1;
  }

  struct sockaddr_in addr;
  addr.sin_family=AF_INET;
  addr.sin_port=htons(port);
  addr.sin_addr = *((struct in_addr *)host->h_addr);
  memset(&(addr.sin_zero), '\0', 8);

  if (::connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    perror("connect");
    return -1;
  }
  return sock;
  //remember to call close(sock) when you're done
}

class BlindPlanner : public Planner
{
private:
  uint plan_pos = 0;
  const std::vector<std::string> actions;
public:
  BlindPlanner(const Problem& problem, const std::vector<std::string> actions)
    : Planner(problem), actions(actions) { }
  ~BlindPlanner() {}
  void initRound() {plan_pos = 0;}
  void endRound() {};
  const Action* decideAction(const AtomSet& atoms,
                             const ValueMap& values) {
    const Action *action = NULL;
    if (plan_pos < actions.size()) {
      ActionList enabled;
      _problem.enabled_actions(enabled, atoms, values);
      std::string action_name = actions[plan_pos];
      for (auto act_p : enabled) {
        // TODO: In some problems, may also need to match on parameters
        if (act_p->name() == action_name) {
          action = act_p;
          break;
        }
      }
      if (!action) {
        std::cout << "WARNING: Couldn't find action with name " << action_name
                  << std::endl;
      }
      plan_pos++;
    } else {
      std::cout << "Have gone through all " << actions.size()
                << " actions; exiting" << std::endl;
    }
    return action;
  }
};

class RandomPlanner : public Planner
{
 public:
  RandomPlanner(const Problem& problem) : Planner(problem) { srand(time(0)); }
  virtual void initRound();
  virtual ~RandomPlanner() {}
  virtual const Action* decideAction(const AtomSet& atoms,
                                     const ValueMap& values);
  virtual void endRound();
};

void RandomPlanner::initRound()
{

}

void RandomPlanner::endRound()
{

}

const Action* RandomPlanner::decideAction(const AtomSet& atoms,
                                          const ValueMap& values)
{
  ActionList actions;
  _problem.enabled_actions(actions, atoms, values);

  if (actions.empty()) {
    return 0;
  }
  else {
    size_t i = size_t(rand()/(RAND_MAX + 1.0)*actions.size());
    return actions[i];
  }
}

void read_actions(char *filename, std::vector<std::string> &out)
{
  std::ifstream in(filename);
  in.exceptions(std::ifstream::badbit);
  while (!in.eof()) {
    std::string line;
    if (!std::getline(in, line)) {
      break;
    }
    if (line.size()) {
      out.push_back(std::move(line));
    }
  }
}

int main(int argc, char **argv)
{
  /* Set default verbosity. */
  verbosity = 0;
  /* Set default warning level. */
  warning_level = 1;
  /* Host. */
  std::string host;
  /* Port. */
  int port = 0;
  /* Actions to take (if applicable) */
  bool plan_randomly = true;
  std::vector<std::string> action_strs;

  try {
    /*
     * Get command line options.
     */
    while (1) {
      int option_index = 0;
      int c = getopt_long(argc, argv, OPTION_STRING,
                          long_options, &option_index);
      if (c == -1) {
        break;
      }
      switch (c) {
      case 'a':
        plan_randomly = false;
        read_actions(optarg, action_strs);
        break;
      case 'H':
        host = optarg;
        break;
      case 'P':
        port = atoi(optarg);
        break;
      case 'v':
        verbosity = (optarg != 0) ? atoi(optarg) : 1;
        break;
      case 'W':
        warning_level = (optarg != 0) ? atoi(optarg) : 1;
        break;
      case 'h':
        display_help();
        return 0;
      case ':':
      default:
        std::cerr << "Try `mdpclient --help' for more information."
                  << std::endl;
        return -1;
      }
    }

    /*
     * Read pddl files.
     */
    if (optind < argc) {
      /*
       * Use remaining command line arguments as file names.
       */
      while (optind < argc) {
        if (!read_file(argv[optind++])) {
          return -1;
        }
      }
    } else {
      /*
       * No remaining command line argument, so read from standard input.
       */
      yyin = stdin;
      if (yyparse() != 0) {
        return -1;
      }
    }

    if (verbosity > 1) {
      /*
       * Display domains and problems.
       */
      std::cerr << "----------------------------------------"<< std::endl
                << "domains:" << std::endl;
      for (Domain::DomainMap::const_iterator di = Domain::begin();
           di != Domain::end(); di++) {
        std::cerr << std::endl << *(*di).second << std::endl;
      }
      std::cerr << "----------------------------------------"<< std::endl
                << "problems:" << std::endl;
      for (Problem::ProblemMap::const_iterator pi = Problem::begin();
           pi != Problem::end(); pi++) {
        std::cerr << std::endl << *(*pi).second << std::endl;
      }
      std::cerr << "----------------------------------------"<< std::endl;
    }

    int socket = connect(host.c_str(), port);
    if (socket <= 0) {
      std::cerr << "Could not connect to " << host << ':' << port << std::endl;
      return 1;
    }

    if (!plan_randomly) {
      std::cout << "Using blind planner with action sequence" << std::endl;
      int i = 0;
      for (auto s : action_strs) {
        std::cout << " [" << i << "]: " << s << std::endl;
        i++;
      }
    } else {
      std::cout << "Using random planer" << std::endl;
    }

    for (Problem::ProblemMap::const_iterator pi = Problem::begin();
         pi != Problem::end(); pi++) {
      RandomPlanner rp(*(*pi).second);
      BlindPlanner bp(*(*pi).second, action_strs);
      Planner *p;
      if (plan_randomly) {
        // choose random action each time
        p = &rp;
      } else {
        // use fixed action sequence from file
        p = &bp;
      }
      XMLClient(*p, *(*pi).second, "johnclient", socket);
    }
  } catch (const std::exception& e) {
    std::cerr << std::endl << "mdpclient: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "mdpclient: fatal error" << std::endl;
    return -1;
  }

  Problem::clear();
  Domain::clear();

  return 0;
}
