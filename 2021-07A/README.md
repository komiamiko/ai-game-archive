# ShipTurret

A 2D continuous-space discrete-time 2-player competitive game involving unit control, resource management, and strategy.

There is 1 map, the same for each game.
Each player begins with 1 home base, which is immobile, and 2 mobile ships.
Game entities do not collide.
Players get a small amount of passive income.
Ships will automatically generate additional income while near mines.
Players can use this currency to build new ships and turrets from their home base, up to a maximum of 20 ships and 5 turrets.
Fog of war limits both players' vision.
Turrets cannot move on their own.
Ships can pick up 1 turret each and drop them.
Turrets cannot shoot while picked up.
Ships are slowed while carrying turrets.
Ships use a single target ranged weapon, which cannot be dodged except by picking up the turret the attack is aiming at.
Turrets use an infinite range piercing weapon, and can shoot in any direction, though they have limited turning speed.
Turrets can only hit ships, while ships can hit anything.
Strategy involves controlling mining locations, managing imperfect information, deception, ship management, and strategic use of turrets.
A match ends in a draw automatically after a very high time limit.

# Requirements

Ideally, you should have

* Linux
* Python 3.9
* A modern browser (for display)

The developers cannot guarantee the game will work on other systems.
