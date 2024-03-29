
# About: Classification of Player moves

A player can make one move of one of the following 
categories per round, in which every round allows for
each player in a `TMEnv` instance (`TM` stands for trap 
match):

- `PMove`
Instructions for executing one of the moves that a player 
is capable of. A `PMove` is a "synchronous" isomorphic
attack from an acting player onto all other players in the
trap-match environment that operates on a shared payoff
rule to incur changes on the relevant player nodes and
edges. There are two main steps, the `PMove`'s effect on
the acting player and the `PMove`'s effect on the other
players.

1. The `PMove` calculates the isomorphisms of its simple
+undirected graph variable "target_resource" on the acting 
player's "ResourceGraph" instance. For every isomorphism 
"I", add the `PMove`'s "payoff" to each node or edge of 
"I".
2. For each of the other players, the `PMove` does virtually
the same calculations as in the first step, except the 
isomorphisms are of the "antitarget_resource" and the delta
is "antipayoff" instead of "payoff".    

- `AMove`
The `AMove`, in which "A" stands for analogue, contains 
instructions that focus on the greatest common subgraph
`C` shared between all players in the trap-match 
environment. It allows an acting player to completely 
destroy all nodes and edges of `C` in each of the other 
players, and evenly distributes the cumulative health of 
all destroyed nodes and edges to the acting player's 
subgraph of its "ResourceGraph". The possible subgraphs
that the acting player can choose for the health transfer
is the subgraph `S_25`, consisting of the nodes and edges 
with health that does not exceed the 25-percentile, or 
the subgraph `S_75`, consisting of the nodes and edges 
with health that does not exceed the 75-percentile.

The greatest common subgraph for the anti-payoff graph of the 
`AMove` is calculated by the `TMEnv` at the start of each 
timestamp. The calculation is done so in a knowledge-agnostic
way, meaning a player's knowledge of another player's 
`ResourceGraph` is not a determinant, so that the greatest common 
subgraph is assigned to each player at the start of a round. In 
order for a player `P_1` to affect another player `P_2` using an 
`AMove`, `P_2` must have an isomorphism of `C` at the time that 
`P_1` executes the `AMove` on it. Exactly 1 or 0 isomorphisms of
an anti-target's resource graphs will be affected by the `AMove`,
and `S_25` xor `S_75` will gain the possible points. 

- `MMove`
The `MMove` is a category of move, in which the "M" stands 
for "modify". There are 3 types of `MMove`s:
"make new nodes and edges", "withdraw" (delete nodes or edges 
and transfer their health to the player's excess/bank), and 
"make new move" (temporary move).

- `NMove`
The "N" stands for negotiation. The concept of 
negotiation is implemented in this program with the 
use of "negochips", and cancellation of "negochips" is 
done using "negachips". Chip objects are tools placed 
upon nodes for one of two effects, nicknamed DISTORT 
and DECEPTION. 

An acting player can place DECEPTION negochips only on 
its own "ResourceGraph". For every node with a 
DECEPTION chip, its outward appearance conceals its 
actual neighbors to other players. DECEPTION chips allow
for a player to hide the real structure of its 
`ResourceGraph` in the event that other players launch 
isomorphic attacks (`PMove`s) against it so that those 
attacks miss their mark. 

An acting player `P` can place DISTORT negochips on its 
own nodes or on another player's nodes. DISTORT uses the
negochip variable `magnitude`, a real number greater than
1.0. If `P` takes a `PMove` that causes a delta to a node,
and the node has a negochip, then one of two things
happen depending on the negochip's owner. If the owner
is `P`, then the delta is multiplied by `magnitude`. 
Otherwise, the delta is divided by `magnitude`.

Negochips have a lifespan of 7 rounds. If a negachip with
a matching is placed on the same node as a negochip at any 
time before its natural termination and the negachip is 
of the same type (DISTORT or DECEPTION), the negochip is 
unnaturally terminated.

# About: TMEnv round 
Every round, players first calculate their contexts and 
then executes their move. Players move in a random ordering
per round.

Every game has two labels for two categories:
- public/private: public mode allows each player more 
information on another player.
- nego/noneg: nego mode allows a player to use `NegoChip` 
and `NegaChip` instances.

## Aspects of the Public/Private Modes
For public mode, the following 
information is provided to each player in its contextual 
calculations before moving:

- the gauging of a `PMove`'s potency allows it to know 
which of the other players' nodes are terminated. See the 
method `potency_default_cumulative_function` for more 
information.
- knowledge of the estimated number of hits nodes of other 
players can take before downed. See the method 
`amove_hitsurvivalrate__other_players`. 
- expected `PMove` isomorphic attack deltas and actual 
`PMove` isomorphic attack deltas, information used by a 
Player to correctly guess negochip locations. See the 
method `Player.register_PMove_hit`.
- other players' ResourceGraph for Negochip type/location 
hypotheses.

* what a player does not know for each round are the payoff gains
made by any other player.

## Aspects of the Nego/NoNeg Modes
- For every player `P`, the negochips that `P` can place are 
constrained by these rules:
* `P` cannot place negochips of type `deception` onto other 
player nodes.
* `P` can place exactly one negochip of one type onto any node. 
For itself, it can place a maximum of two negochips, one of 
`deception` and one of `distortion`, onto its nodes. For any 
node of another player, it can place only one negochip of 
`distortion` on it.

## Context Structure for Player Decision
The data structure that a Player uses to conduct decisions at
every timestamp is the `PContext` structure, which provides 
information for the possible moves it can take, each of one of
the types `PMove`,`AMove`,`MMove`, and `NMove`. Every 
possible `XMove` has an `XInfo` counterpart that stores 
important information on it. The `XInfo` data structures are
what is stored by `PContext`. When a player is to use a 
`PContext` instance to make a decision, the `PContext` 
outputs information according to a particular structure that 
complies with the format used by `StdDecFunction`, a structure
that uses adjustable linear combinative weights to output 
singular float values for every possible decision. The 
greatest float value corresponds to the decision that the 
player will take for the timestamp.

The format of the `StdDecFunction` weights is as following:
```
[0] PInfo weights vec, size 11
[1] AInfo weights vec #1, size 7
[2] AInfo weights vec #2, size 7
[3] MInfo weights vec #1, size 3
[4] MInfo weights vec #2, size 5
[5] NInfo weights vec #1, size 2
``` 

NOTE: the size of the weights above excludes the error terms 
used.

The variables for each of the `XInfo` weights are the 
following:
- `PInfo`: information about a player's PMove onto 
another player (could be onto self). Every gauge of
a (`PMove`, `Player`) pair produces an instance.  
    * node potency, continuous (magnitude of predicted float delta)
    * edge potency, continuous (magnitude of predicted float delta)
    * node potency, boolean (mortality by predicted float delta); accurate only in public information mode.
    * edge potency, boolean (mortality by predicted float delta); accurate only in public information mode. 
    * size of number of node additions for an additional isomorphism.
    * size of number of edge additions for an additional isomorphism.
    * mean of node frequency from isomorphic attack
    * minumum of node frequency from isomorphic attack
    * maximum of node frequency from isomorphic attack
    * cumulative health delta of nodes
    * cumulative health delta of edges
- `AInfo`: two possible `AMoves` that can be taken by 
a player, given a calculated greatest common 
subgraph, is the 25th percentile `AMove` or 75th percentile `AMove`. 
    * expected gains for owner's payoff target.
    * minimum hit survival rate of (25|75)'th percentile graph
    * maximum hit survival rate of (25|75)'th percentile graph
    * mean expected losses of players 
    * mean ratio of nodes+edges lost for players if `AMove` is successfully executed
    * mean minimum hit survival rate of greatest common subgraph for other players
    * mean maximum hit survival rate of greatest common subgraph for other players
- `MInfo`: program supports the two `MMove` types 
"make new nodes + edges" and "withdraw". <ins>The other type, "make new move" is currently not supported.</ins>
The variables for "make new nodes + edges" (#1) are
    * c1: see the function `player.mmove_addition_score_move__type_1`  
    * c2: see the function `player.mmove_addition_score_move__type_1`  
    * c3: see the function `player.mmove_addition_score_move__type_1`  
For a player of `n` moves that it has information on, there will be
`n` possible `MInfo#1` types to choose from. The variables for 
"withdraw" are  
    * number of 1-hit nodes belonging to actor  
    * number of 1-hit edges belonging to actor  
    * number of 2-hit nodes belonging to actor  
    * number of 2-hit edges belonging to actor   
    * minimum hit survival rate
- `NInfo`: 
    * 0 for `NegaChip` move, 1 for `Negochip` move  
    * expected cumulative delta from `NMove`  

# The <DefInt> Structure in Player Decision-Making
There is a structure used by a player for defensive 
intelligence, deemed `DefInt`. Its variables of
pertinence are 

- changes inflicted by other players onto self  
    * node identifier -> other player idn. -> <negative delta sequence>  
    * edge identifier -> other player idn. -> <negative delta sequence>  
    * node identifier -> estimated number of hits to down
    * edge identifier -> estimated number of hits to down
- information on self's `PMove` onto others  
    * pmove -> player -> node/edge -> expected,actual
- information on self's `PMove` onto self
    * pmove -> node/edge -> expected,actual

`DefInt` was conceptualized with minimalism as the
objective, so it contains only a handful of relevant 
variable classes. `DefInt` logs the effects of `PMove`s
of other players, stores the estimated survivability of
nodes and edges using the metric "estimated number of
hits to down", and stores the (expected,actual) deltas
of `PMove`s onto others and self.

These variable categories aid in calculating values
of pertinence for the `PContext` structure.

# The PKDB structure

The structure `PKDB` is an acronym for Player Knowledge 
Database. Each player possesses an instance of this structure.
At every timestamp, before a player decides on its best move,
it uses the knowledge of other player's `MicroGraph` instances
in the `PKDB` to gauge the score of every possible move that
involves another player's `ResourceGraph`. 

A player is able to collect knowledge of another player's `MicroGraph`
through its two unit moves (see `sample_resource_graph_6` and 
`sample_resource_graph_7`). These moves are deemed the unit moves because they 
together are able to accurately predict the structure of other `MicroGraph` 
instances through their `PMove` execution. These two moves are the only
moves that adds knowledge of other player's `MicroGraph` structures to
a player's `PKDB`. 

Of important note is the `AMove` does not require knowledge from the 
`PKDB` to execute. 

# The PDEC structure

This is the deciding structure of the player. It is responsible for 
calculating the `PContext` at every timestamp for the player. It also
holds the player's `PKDB` and `DefInt` instances. Of another importance 
is its responsibility in calculating suspected negochips by other players 
based on the differences between expected and actual scores of `PMove` executions by the player. 

# The Player Move-Chain

Player gauges each possible move by the 
following order of categories: 
- `PMove`  
- `AMove`
- `MMove`
- `NMove`

For each possible move of category `X`, the 
player generates an `XInfo` instance that 
numerically captures the forecasted effects of
that move based on the knowledge that player 
has at that time of forecasting, held in its
`PDEC` structure. Then the player uses the 
standard deciding function `F` to rank the 
possible moves by the numerical values 
collected. Lastly, the player chooses the 
move with the highest score calculated by 
`F`.

# The Learning Process
- data collection phase
The structure `FARSE` takes one `TMEnv` instance. User 
inputs the target player in the `TMEnv` for `FARSE` to 
record its information for training. Until the player 
wins (by being the only surviving player in the `TMEnv`) 
or is terminated, `FARSE` uses a branching decision 
structure (similar to a decision tree that grows larger 
with every timestamp) to determine the best sequences of
moves conducted by the player at every hop sequence junction.
Relevant information in the form of the 24-vector (variable 
values collected from a `PContext` instance of the player) 
and the `PContext` at best-decision junction points is  
collected into a folder. 
- learning phase 
The structure `ZugdeinlichtMach` takes as input a sequence 
of folder information, each pertaining to the information 
collected from a `FARSE` training instance.  

`ZugdeinlichtMach` then uses a modified form of the `BallComp`
unsupervised machine-learning structure (in the library `morebs2`)
to classify the 24-vecs so that a `BallComp` classifier solution 
is constructed. `ZugdeinlichtMach` then iterates back through 
the 24-vecs and collects frequency labels for  
`ball identifier -> XMove class for the 24-vec`.
`ZugdeinlichtMach` goes back and iterates through each (24-vec, `PContext`)
pair and calculates the sequence of weights for the `PDEC` structure to 
use so that the wanted decision from `PContext` is used. 

# Miscellaneous features
- `move-type deteterministic`
description:  
pseudo-random decision-making mechanism (uses Python 
random seeds) for a Player that exclusively chooses a 
move of a specified set of types. If no moves exist of 
those types, then chooses any arbitrary move.

uses:  
aids in testing the performance of specific moves 
executed by players.  

- `preferred move`
description:
feature of the class `TMEnv` used to assign players the
decision of `preferred move` in making their decisions
for each round. 

uses: 
aids in testing the performance of specific moves
executed by players. 