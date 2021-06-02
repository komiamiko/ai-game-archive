# Basic setup

You will need to set up your bot so it can be run as an executable.

Your bot will be instantiated once at the start of each match, and terminated after the match.

At the start of each tick, your bot will receive input through standard input.

Your bot should output all actions to standard output.

At the end of the match, your bot should terminate gracefully.

In all tournament phases other than the opening phase, multiple matches may be running at the same time, including multiple matches using the same bot.
Please make sure your bot is safe to run in parallel with other instances of itself.

Using information from other matches within the tournament is considered cheating, and not allowed.
Using information from past matches done offline, for training for example, is okay.

Failing to meet these requirements will result in your bot being disqualified.

As courtesy, please also make your bot reasonably fast.
The runner script is not guaranteed to enforce timeouts.
There is diminishing returns for more computation past a certain point, and it is recommended to improve your bot by making it smarter about strategy rather than giving it more time to think.

## Recommended languages

Python is a good well rounded language.
Scripts can be run directly with ex. `python3 bot.py`

C++ is good for high performance.
You could use a build tool like CMake within your preparation commands, and then use the executable directly as your run bot command.

There are plenty of other good languages out there, like Rust as an alternative for high-ish level high performance code.
Any other language which can be run directly in this manner is also fair game.

## Tournament phases

The tournament consists of 3 phases.
It reverts to a simpler mode if there are not enough bots to warrant the more complex behaviour.

### Opening phase

The main purpose of this phase is to initialize all bots.

There are not many matches.

Each bot is guaranteed to be included in one match.

Matches are not run in parallel, to ensure initialization of all bots is done correctly.

### Main phase

In the main phase, the match scheduler will continuously match bots of similar ratings, run matches, and update the ratings.
This will occur for many rounds, according to tournament parameters.
This should establish ratings for all bots.

The main phase is the longest phase, and has the highest parallelism factor.

### Finals phase

The finals phase is meant to determine the winners and the ratings of the top bots.
It is conceptually similar to the main phase, except the match scheduler heavily favours picking bots within the top few, round are smaller, and the parallelism factor is not as high.

Some sample games will also be saved.
These may be interesting for humans to look at and analyze.

# General structure of a tick

* Query both players
* Home bases do ship, turret production
* Ships, turrets do attack
* Ships do turret drop
* Ships do turret pick up
* Ships, turrets, attacks do movement
* Attacks do damage
* Dead units removed
* Generate income
* Increment timers

# Input specification

One per line.

The convention for players is 0 = you, 1 = opponent.

Positions are in game space units.
Angles are in radians.

Unit IDs are globally unique and never reused within a match.
ID 0 indicates no unit or information not available.
IDs are pseudorandomly generated, and always fit in an unsigned 63-bit integer, which means they also fit in a signed 64-bit integer.
Please do not use IDs to gain information on opponent production.

## End of match

`end_game`

Indicates the match is over.
Player program should terminate gracefully.

## End of tick info

`end_tick`

This will be the last line for each tick.

## Current tick

`tick <number>`

Number of ticks elapsed since start of game.
Will be 0 on the first tick.

## Self crystals

`currency <number>`

Current amount of resources you have.

## Mine

`mine <x> <y>`

Location of a mine.

## Home base

`home <id> <player> <x> <y> <hp>`

Home base.

## Ship

`ship <id> <player> <x> <y> <hp> <vx> <vy> <reload> <turret_id>`

Mobile ship.

Velocity is in space units per tick.

Reload is the number of ticks before it can shoot again.
If it is 0, the ship is ready to shoot.

Turret ID is the ID of the currently picked up turret, if any.
Opponent picked up turrets are not visible, so for opponent ships, this will always be 0.

## Turret

`turret <id> <player> <x> <y> <hp> <facing> <reload>`

Facing is the angle it is facing.

## Ship attack

`ship_attack <target_id> <delay>`

Note that the attack does not have a position.
Positions only exist for visualization purposes.

The delay is the number of ticks until it hits.
If it is 0, it is about to hit.

## Turret attack

`turret_attack <player> <x> <y> <vx> <vy>`

Turret attacks will only hit opponent ships.

# Output specification

One per line.

In general, instructions that cannot be carried out will be ignored, though the game runner will attempt to execute all of your instructions, in order.

## End of tick actions

`end_action`

This must be the last line for each tick.

## Produce ship

`make_ship`

## Produce turret

`make_turret`

## Move ship

`move_ship <id> <target_vx> <target_vy>`

Velocity is limited by maximum velocity.
Also, change in velocity is limited by maximum acceleration.

## Ship attack

`ship_attack <target_id>`

You are allowed to attack your own ships.
This may be useful to prune low HP ships.

## Pick up turret

`pickup <ship_id> <turret_id>`

## Drop turret

`drop <ship_id>`

## Rotate turret

`turret_aim <target_facing>`

Change in facing angle is limited by maximum turning speed.

## Turret attack

`turret_attack`

Turret shots always move at full speed.
You cannot control this.

# Further details

For precise game details, see `run_match.py`.
This will be a useful reference for developing your bot.
You may want to copy important constants.