"""
Rush K bot
* gather crystals at middle mine
* build ships
* at a certain ship count, switch to all in attack
* attack all enemies on sight with random targeting
"""

import math
import random
import sys

MINE_LOCATIONS = [(0,-4),(0,1),(-2,6),(2,6),(-6,10),(6,10)]
SHIP_ATTACK_RANGE = 1

attack_k = int(sys.argv[1])
which_mine = int(sys.argv[2])
main_mine = MINE_LOCATIONS[which_mine]

ships = []
enemies = []
enemy_base = [0, 0]

while True:
    s = input()
    if s == 'end_game':
        break
    if s == 'end_tick':
        print('make_ship')
        for sid, sx, sy in ships:
            targets = []
            for eid, ex, ey in enemies:
                if math.hypot(ex - sx, ey - sy) <= SHIP_ATTACK_RANGE:
                    targets.append(eid)
            if targets:
                tid = random.choice(targets)
                print(f'ship_attack {sid} {tid}')
            if len(ships) >= attack_k:
                tx, ty = enemy_base
            else:
                tx, ty = main_mine
                if enemy_base[0] > 0:
                    tx = -tx
            tvx = tx - sx
            tvy = ty - sy
            print(f'move_ship {sid} {tvx} {tvy}')
        print('end_action')
        ships = []
        enemies = []
    else:
        command, *xargs = s.split()
        if command == 'home':
            sid = int(xargs[0])
            player = int(xargs[1])
            x = float(xargs[2])
            y = float(xargs[3])
            if player == 0:
                enemy_base = -x, y
            else:
                enemy_base = x, y
                enemies.append((sid, x, y))
        elif command == 'ship':
            sid = int(xargs[0])
            player = int(xargs[1])
            x = float(xargs[2])
            y = float(xargs[3])
            if player == 0:
                ships.append((sid, x, y))
            else:
                enemies.append((sid, x, y))
        elif command == 'turret':
            sid = int(xargs[0])
            player = int(xargs[1])
            x = float(xargs[2])
            y = float(xargs[3])
            if player == 0:
                pass
            else:
                enemies.append((sid, x, y))
