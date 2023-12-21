# r2apart

Synchronicity of variable networks based on shared payoff rules in multi-agent games. 

`UNFINISHED DEVELOPMENT OF GAME`.

# Running the tests
python3 -m unittest discover tests
https://stackoverflow.com/questions/1732438/how-do-i-run-all-python-unit-tests-in-a-directory

# Broad Overview
Development of project has hit a "brick 
wall". The fundamental NP-Complete problem,
subgraph isomorphism, which is the primary
move (PMove) used by players to enhance their
resource graph healths or damage the resource
graph healths of other players, is too 
expensive to calculate for each timestamp 
that players are to gauge or execute PMoves.

As a result of the expensive calculations
revolving around the subgraph isomorphism
problem, as evidenced by the long runtimes
(around 40 minutes on a typical personal 
computer for a game of three players in the 
first timestamp, each with graph degrees 
between 16 and 30), there is little
practical motivation to continue development 
of the game, `r2apart`, so that the game 
execution is completed (bug-free) and the
players have machine-learning capabilities.

Development of game serves as a detour from
studies on the NP-Complete problem, 
subgraph isomorphism, into programmatic
implementation and practical uses.

As of now, no known practical uses of the
existing codebase for this game can be
thought of by its author.

Please see the file `technicals` for a brief
overview of the types of moves (`PMove`,
`AMove`,`MMove`,`NMove`) that a player can
take for each timestamp.

# License?
Copyright 2023 Richard Pham.