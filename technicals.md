
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

The `AMove` takes two rounds to complete. The first round
is the planning stage, and the second round is the 
execution stage. In order for a player `P_1` to affect 
another player `P_2` using an `AMove`, `P_2` must have an 
isomorphism of `C` at the time that `P_1` executes the 
`AMove` on it.

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