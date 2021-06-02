"""
Rush 1 bot
* send starting ships directly to the enemy base
* only attack the enemy base
* produces new ships
"""

enemy_base = [None, 0, 0]
ships = []

while True:
    s = input()
    if s == 'end_game':
        break
    if s == 'end_tick':
        eid, ex, ey = enemy_base
        for sid, x, y in ships:
            vx = ex - x
            vy = ey - y
            print(f'move_ship {sid} {vx} {vy}')
            if eid is not None:
                print(f'ship_attack {sid} {eid}')
        print('make_ship')
        print('end_action')
        ships = []
    else:
        command, *xargs = s.split()
        if command == 'home':
            sid = int(xargs[0])
            player = int(xargs[1])
            x = float(xargs[2])
            y = float(xargs[3])
            if player == 0:
                enemy_base[1] = -x
                enemy_base[2] = y
            elif player == 1:
                enemy_base = [sid, x, y]
        elif command == 'ship':
            sid = int(xargs[0])
            player = int(xargs[1])
            x = float(xargs[2])
            y = float(xargs[3])
            if player == 0:
                ships.append([sid, x, y])