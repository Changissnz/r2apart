# r2apart

Synchronicity of variable networks based on shared payoff rules in multi-agent games. 

`UNFINISHED DEVELOPMENT OF GAME`.

# Running the tests
- all the tests 
`python3 -m unittest discover tests`  
https://stackoverflow.com/questions/1732438/how-do-i-run-all-python-unit-tests-in-a-directory
- single files
`python3 -m unittest tests/FILENAME.py`


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
(around 40 minutes for the first timestamp 
on a typical personal computer in a game of 
three players, each with graph degrees 
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

# Update: 1/2/2024

Decided to continue on with work on this project. 
Downsides of expensive computational costs of subgraph
isomorphism resolved in terms of time with the use of
the global variable `DEFAULT_ISOMORPHIC_ATTACK_SIZE`.

# Update: 1/5/2024 

Still bugs in `TMEnv.tme.move_one_timestamp` as 
well as slow execution time. However, the program
is progressing ...

# Update: 1/6/2024
note#1
------
- Trimmed down the runtime for the
1st timestamp to approximately 385 
seconds.

note#2
------
- Trimmed down the runtime for the 
first 5 timestamps to under 2 minutes.

# Update: 1/30/2024

- The `FARSE` training algorithm is well 
underway in construction.

# Update: 2/5/2024 

- Finalized design plans for project. Project has
not yet implemented `MMove #3` (make new move). 
Perhaps `MMove #3` will not be implemented.
- Start work using `class<BallComp>` from library 
`morebs2`.

# Update: 2/6/2024

- `MMove #3` will not be implemented due to 
computational shortcomings on standard computing 
devices that have resulted in slow runtimes for 
the player move chain as it already is.

# Update: 2/18/2024

- Taking a < 1 month break from this project. 
  Further developments stalled until after break. 

# Update: 2/27/2024

- Development has slowed to a snail's crawl. The 
file `farse_mach_samples` has code for two `class<FarseMach>` instances. 
Please see the file `technicals.md` for the data collection and 
training procedure. 
- Many sub-optimal parts in this program, with possible bugs present 
in the program. 

## ATTENTION 
Public developments for this project are terminated due to lack of 
  time and resources on the developer's part (that would be me). 
The specific parameters involved in this automaton demand very 
specific code, and the runtime as well as commercial costs involved 
in `r2apart` have encouraged me to cease all developments for it made 
public. 

# License?
Copyright 2023 Richard Pham.