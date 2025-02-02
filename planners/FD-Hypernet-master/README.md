This fork adapts Fast Downward to be able to use hypergraph networks as a heuristic function.

Requirements:

- pytorch >= 1.5.0
- python-3.7
- python3-dev package
- Access to https://gitlab.com/williamshen/strips-hgn or https://gitlab.com/geisserf/strips-hgn

Environment variables:
- FD_HGN: path to this repository
- STRIPS_HGN_NEW: path to strips-hgn path

Installation:
 - Clone this repository
 - clone https://gitlab.com/geisserf/strips-hgn to /path/to/hgn and
   setup local Fast Downward repository:
   `clone https://github.com/williamshen-nz/fast_downward.git /path/to/hgn/src/fast_downward`

 - setup FD_HGN and STRIPS_HGN_NEW environment variables
 - ./build.py

 
Usage:
Make sure that /path/to/hgn is on $PYTHONPATH.

In the following, we assume you want to evaluate instance_k of domain d and
there already exists trained network. We denote with INSTANCE, DOMAIN and
NETWORK the full (or relative to current working directory) path to the
instance, domain and network file, respectively.

```
./fast-downward.py  DOMAIN INSTANCE  --translate-options --full-encoding --search-options --search "astar(hgn2(network_file=NETWORK, domain_file=DOMAIN, instance_file=INSTANCE))"
```

Optionally you can use the `num_steps=k` argument for integer k>0 to change the
message passing step size.

Example:

In the following /path/to/hgn is ../strips-hgn/ and we will set
- DOMAIN to `benchmarks/blocksworld/domain.pddl`
- INSTANCE to `benchmarks/blocksworld/blocks3_task23.pddl`
- NETWORK to `weights/blocksworld/model-best-0.ckpt`

We can then run Fast Downward using the HGN heuristic with the following command:

```
./fast-downward.py benchmarks/blocksworld/domain.pddl benchmarks/blocksworld/blocks3_task23.pddl  --translate-options --full-encoding --search-options --search "astar(hgn2(network_file=weights/blocksworld/model-best-0.ckpt, domain_file=benchmarks/blocksworld/domain.pddl, instance_file=benchmarks/blocksworld/blocks3_task23.pddl))"
```

The following content is the original Fast Downward README.
=======



Fast Downward is a domain-independent planning system.

Copyright 2003-2019 Fast Downward contributors (see below).

For documentation and contact information see
<http://www.fast-downward.org>.


Contributors
============

The following list includes all people that actively contributed to
Fast Downward, i.e. all people that appear in some commits in Fast
Downward's history (see below for a history on how Fast Downward
emerged) or people that influenced the development of such commits.
Currently, this list is sorted by the last year the person has been
active, and in case of ties, by the earliest year the person started
contributing, and finally by last name.

- 2003-2019 Malte Helmert
- 2008-2016, 2018-2019 Gabriele Roeger
- 2010-2019 Jendrik Seipp
- 2010-2011, 2013-2019 Silvan Sievers
- 2012-2019 Florian Pommerening
- 2013, 2015-2019 Salome Eriksson
- 2015-2019 Manuel Heusner
- 2016-2019 Cedric Geissmann
- 2017-2019 Guillem Francès
- 2018-2019 Augusto B. Corrêa
- 2018-2019 Patrick Ferber
- 2017 Daniel Killenberger
- 2016 Yusra Alkhazraji
- 2016 Martin Wehrle
- 2014-2015 Patrick von Reth
- 2015 Thomas Keller
- 2009-2014 Erez Karpas
- 2014 Robert P. Goldman
- 2010-2012 Andrew Coles
- 2010, 2012 Patrik Haslum
- 2003-2011 Silvia Richter
- 2009-2011 Emil Keyder
- 2010-2011 Moritz Gronbach
- 2010-2011 Manuela Ortlieb
- 2011 Vidal Alcázar Saiz
- 2011 Michael Katz
- 2011 Raz Nissim
- 2010 Moritz Goebelbecker
- 2007-2009 Matthias Westphal
- 2009 Christian Muise


History
=======

The current version of Fast Downward is the merger of three different
projects:

- the original version of Fast Downward developed by Malte Helmert
  and Silvia Richter
- LAMA, developed by Silvia Richter and Matthias Westphal based on
  the original Fast Downward
- FD-Tech, a modified version of Fast Downward developed by Erez
  Karpas and Michael Katz based on the original code

In addition to these three main sources, the codebase incorporates
code and features from numerous branches of the Fast Downward codebase
developed for various research papers. The main contributors to these
branches are Malte Helmert, Gabi Röger and Silvia Richter.


License
=======

The following directory is not part of Fast Downward as covered by
this license:

- ./src/search/ext

For the rest, the following license applies:

```
Fast Downward is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

Fast Downward is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```
