"""
Runs a single match between 2 bots, reports the result, then exits.
"""

import argparse
import itertools
import math
import pathlib
import random
import subprocess
import typing
from .bot_info import BotInfo, get_all_bots

# --- game related constants ---

BASE_LOCATIONS = [(-10,0),(10,0)]
MINE_LOCATIONS = [(-3,1),(3,1),(0,4),(0,-2),(0,-6)]
TYPE_BASE = 0
TYPE_SHIP = 1
TYPE_TURRET = 2
TYPE_TURRET_ATTACK = 3
BASE_HP = 1000
SHIP_STARTING_N = 2
SHIP_LIMIT_N = 20
SHIP_LIMIT_V = 0.1
SHIP_LIMIT_V_CARRY = SHIP_LIMIT_V * 0.5
SHIP_LIMIT_A = SHIP_LIMIT_V * 0.5
SHIP_ATTACK_RANGE = 1
SHIP_ATTACK_RELOAD = 5
SHIP_ATTACK_DELAY = 4
SHIP_ATTACK_DAMAGE = 5
SHIP_HP = 30
TURRET_LIMIT_N = 5
TURRET_LIMIT_T = math.pi / 25
TURRET_INITIAL_FACING = math.pi / 2
TURRET_ATTACK_V = 0.2
TURRET_ATTACK_R = 0.35
TURRET_ATTACK_RELOAD = 10
TURRET_ATTACK_DAMAGE = 3
TURRET_ATTACK_PRUNE_MARGIN = 2.5
TURRET_HP = 50
PICKUP_RANGE = 0.1
BASE_VISION_RANGE = 2
SHIP_VISION_RANGE = 1
TURRET_VISION_RANGE = 0.7
MINE_RANGE = 0.5
PASSIVE_INCOME = 10
MINE_INCOME = 2
SHIP_COST = 1000
TURRET_COST = 3000
TICK_LIMIT = 10000

# --- actual program ---

class GameEntity(object):
    """
    Represents a game entity.
    """
    def __init__(self, player: int, vid: int, vtype: int, x: float, y: float,
        hp: int, vision: float, reload: int = 0, vx: float = 0, vy: float = 0, f: float = 0,
        ax: float = 0, ay: float = 0, vf: float = 0, carrying: int = 0):
        self.player = player
        self.vid = vid
        self.vtype = vtype
        self.x = x
        self.y = y
        self.hp = hp
        self.vision = vision
        self.reload = reload
        self.vx = vx
        self.vy = vy
        self.f = f
        self.ax = ax
        self.ay = ay
        self.vf = vf
        self.carrying = carrying
    def movement(self):
        self.vx += self.ax
        self.vy += self.ay
        self.ax = 0
        self.ay = 0
        self.x += self.vx
        self.y += self.vy
        self.f += self.vf
        self.vf = 0
        self.f %= 2 * math.pi
    def distance(self, other):
        if isinstance(other, GameEntity):
            return math.hypot(self.x - other.x, self.y - other.y)
        return math.hypot(self.x - other[0], self.y - other[1])

class MatchResult(object):
    """
    Represents the result of a match.

    outcome can be:
    * 1 for player 1 win
    * 0 for draw
    * -1 for player 2 win

    reason can be:
    * OK a player's home base is destroyed, or the tick limit is reached
    * INITIALIZATION_FAIL a bot failed during initialization
    * TIMEOUT a bot exceeded the hard time limit per tick
    * DIED_EARLY a bot process died before the match was over
    * INVALID_ACTION a bot issued an invalid instruction
    """
    def __init__(self, outcome: int, reason: str, ticks: int):
        self.outcome = outcome
        self.reason = reason
        self.ticks = ticks
    def __str__(self):
        return str(self.outcome) + '\n' + self.reason + '\n' + str(self.ticks)

def parse_xargs(xargs: list[str], types: list[type]) -> typing.Optional[list[typing.Union[int, float]]]:
    """
    Parse the arguments according to the expected types.
    If the number of arguments is wrong or the parsing fails, returns None.
    Otherwise, returns the parsed values.
    """
    if len(xargs) != len(types):
        return None
    result = []
    for xs, ty in zip(xargs, types):
        try:
            xv = ty(xs)
            result.append(xv)
        except ValueError:
            return None
    return result

class HalfState(object):
    """
    Represents the side of the game state corresponding to one player.
    2 such halves form the whole game state.

    This class exists more to reduce code duplication than anything else.
    """
    def __init__(self, player: int, bot_info: BotInfo, id_gen: typing.Generator[int, None, None]):
        self.player = player
        self.bot_info = bot_info
        self.bot_sub = None
        self.id_gen = id_gen
        self.units = {}
        self.base = None
        self.ship_attacks = []
        self.turret_attacks = []
        self.currency = 0
        self.tick_counter = 0
        self.other_half = None
    def link(self, other: 'HalfState'):
        """
        Link this half with the other half.
        """
        self.other_half = other
    def add_unit(self, unit: GameEntity):
        self.units[unit.vid] = unit
        if unit.vtype == TYPE_BASE:
            self.base = unit.vid
    def player_alive(self) -> bool:
        """
        Is the player still alive in the match?
        """
        return self.base in self.units
    def startup(self) -> typing.Optional[str]:
        """
        Attempt to initialize and start up the bot program.

        Return None for success, or a fail reason for failure.
        """
        bot_info = self.bot_info
        for command in bot_info.commands[:-1]:
            sub = subprocess.Popen(command)
            ret = sub.wait()
            if ret:
                return 'INITIALIZATION_FAIL'
        command = bot_info.commands[-1]
        self.bot_sub = subprocess.Popen(command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        return None
    def shutdown(self):
        """
        Terminate the bot and release any other resources.
        """
        bot_sub = self.bot_sub
        if bot_sub is not None:
            ret = bot_sub.poll()
            if ret is None:
                try:
                    bot_sub.stdin.write('end_game\n')
                    bot_sub.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                ret = bot_sub.poll()
                if ret is None:
                    bot_sub.kill()
        self.other_half = None
    def set_starting_state(self):
        """
        Set the half state to be the starting game state.
        """
        player = self.player
        id_gen = self.id_gen
        self.add_unit(GameEntity(player, next(id_gen), TYPE_BASE,
            BASE_LOCATIONS[player][0], BASE_LOCATIONS[player][1], BASE_HP, BASE_VISION_RANGE))
        for _ in range(SHIP_STARTING_N):
            self.add_unit(GameEntity(player, next(id_gen), TYPE_SHIP,
                BASE_LOCATIONS[player][0], BASE_LOCATIONS[player][1], SHIP_HP, SHIP_VISION_RANGE))
    def step_query(self) -> typing.Optional[str]:
        """
        Query the bot player, and retrieve the next set of actions.

        Return None for success, or a fail reason for failure.
        """
        # load locals
        player = self.player
        if player == 0:
            fix_player = lambda i:i
        else:
            fix_player = lambda i:1-i
        bot_sub = self.bot_sub
        ohalf = self.other_half
        sunits = self.units
        ounits = ohalf.units
        sship_attacks = self.ship_attacks
        oship_attacks = ohalf.ship_attacks
        sturret_attacks = self.turret_attacks
        oturret_attacks = ohalf.turret_attacks
        currency = self.currency
        tick_counter = self.tick_counter
        # determine visible elements
        visible_bases = {}
        visible_ships = {}
        visible_turrets = {}
        visible_ship_attacks = []
        visible_turret_attacks = []
        masked_turrets = set()
        for unit in sunits.values():
            if unit.vtype == TYPE_BASE:
                visible_bases[unit.vid] = unit
            elif unit.vtype == TYPE_SHIP:
                visible_ships[unit.vid] = unit
            elif unit.vtype == TYPE_TURRET:
                visible_turrets[unit.vid] = unit
        for ounit in ounits.values():
            if unit.vtype == TYPE_SHIP:
                masked_turrets.add(unit.carrying)
        for ounit in ounits.values():
            seen = False
            for sunit in sunits.values():
                if sunit.distance(ounit) <= sunit.vision:
                    seen = True
                    break
            if seen:
                if ounit.vtype == TYPE_BASE:
                    visible_bases[ounit.vid] = ounit
                elif ounit.vtype == TYPE_SHIP:
                    visible_ships[ounit.vid] = ounit
                elif ounit.vtype == TYPE_TURRET and ounit.vid not in masked_turrets:
                    visible_turrets[ounit.vid] = ounit
        for attack in itertools.chain(sship_attacks, oship_attacks):
            delay, target_id = attack
            if target_id in visible_ships or target_id in visible_turrets or target_id in visible_bases:
                visible_ship_attacks.append(attack)
        for attack in sturret_attacks:
            visible_turret_attacks.append(attack)
        for attack in oturret_attacks:
            seen = False
            for unit in sunits:
                if unit.distance(attack) <= unit.vision:
                    seen = True
                    break
            if seen:
                visible_turret_attacks.append(attack)
        # construct input
        unit_key = lambda unit:(fix_player(unit.player), unit.vid)
        lines = []
        lines.append(f'tick {tick_counter}')
        lines.append(f'currency {currency}')
        for mx, my in MINE_LOCATIONS:
            lines.append(f'mine {mx} {my}')
        for unit in sorted(visible_bases.values(), key=unit_key):
            lines.append(f'home {unit.vid} {fix_player(unit.player)} {unit.x} {unit.y} {unit.hp}')
        for unit in sorted(visible_ships.values(), key=unit_key):
            lines.append(f'ship {unit.vid} {fix_player(unit.player)} {unit.x} {unit.y} {unit.hp}' + \
                f' {unit.vx} {unit.vy} {unit.reload} {unit.carrying}')
        for unit in sorted(visible_turrets.values(), key=unit_key):
            lines.append(f'turret {unit.vid} {fix_player(unit.player)} {unit.x} {unit.y} {unit.hp}' + \
                f' {unit.f} {unit.reload}')
        for delay, target_id in visible_ship_attacks:
            lines.append(f'ship_attack {target_id} {delay}')
        for unit in sorted(visible_turret_attacks, key=unit_key):
            lines.append(f'turret_attack {fix_player(unit.player)} {unit.x} {unit.y} {unit.vx} {unit.vy}')
        lines.append('end_tick\n')
        raw_in = '\n'.join(lines)
        # send input, get output
        bot_sub.stdin.write(raw_in)
        actions_production = []
        actions_move = []
        actions_ship_attack = []
        actions_pickup = []
        actions_turret_turn = []
        actions_turret_attack = []
        while True:
            ret = bot_sub.poll()
            if ret is not None:
                return 'BOT_DIED'
            action_line = bot_sub.stdout.readline().strip()
            if action_line == 'end_action':
                break
            if action_line == '':
                continue
            command, *xargs = action_line.split()
            if command == 'make_ship':
                parsed_args = parse_xargs(xargs, [])
                if parsed_args is None:
                    return 'INVALID_ACTION'
                actions_production.append(TYPE_SHIP)
            elif command == 'make_turret':
                parsed_args = parse_xargs(xargs, [])
                if parsed_args is None:
                    return 'INVALID_ACTION'
                actions_production.append(TYPE_TURRET)
            elif command == 'move_ship':
                parsed_args = parse_xargs(xargs, [int, float, float])
                if parsed_args is None:
                    return 'INVALID_ACTION'
                actions_move.append(parsed_args)
            elif command == 'ship_attack':
                parsed_args = parse_xargs(xargs, [int, int])
                if parsed_args is None:
                    return 'INVALID_ACTION'
                actions_ship_attack.append(parsed_args)
            elif command == 'pickup':
                parsed_args = parse_xargs(xargs, [int, int])
                if parsed_args is None:
                    return 'INVALID_ACTION'
                actions_pickup.append((True, parsed_args))
            elif command == 'drop':
                parsed_args = parse_xargs(xargs, [int])
                if parsed_args is None:
                    return 'INVALID_ACTION'
                actions_pickup.append((False, parsed_args))
            elif command == 'turret_aim':
                parsed_args = parse_xargs(xargs, [int, float])
                if parsed_args is None:
                    return 'INVALID_ACTION'
                actions_turret_turn.append(parsed_args)
            elif command == 'turret_attack':
                parsed_args = parse_xargs(xargs, [int])
                if parsed_args is None:
                    return 'INVALID_ACTION'
                actions_turret_attack.append(parsed_args)
            else:
                return 'INVALID_ACTION'
        # record actions
        self.actions_production = actions_production
        self.actions_move = actions_move
        self.actions_ship_attack = actions_ship_attack
        self.actions_pickup = actions_pickup
        self.actions_turret_turn = actions_turret_turn
        self.actions_turret_attack = actions_turret_attack
    def step_production(self):
        """
        Production step.
        Attempt to produce new ships or turrets.
        """
        player = self.player
        id_gen = self.id_gen
        num_ships = 0
        num_turrets = 0
        for sunit in self.units.values():
            if sunit.vtype == TYPE_SHIP:
                num_ships += 1
            elif sunit.vtype == TYPE_TURRET:
                num_turrets += 1
        for action in self.actions_production:
            if action == TYPE_SHIP:
                if self.currency >= SHIP_COST and num_ships < SHIP_LIMIT_N:
                    self.currency -= SHIP_COST
                    num_ships += 1
                    self.add_unit(GameEntity(player, next(id_gen), TYPE_SHIP,
                        BASE_LOCATIONS[player][0], BASE_LOCATIONS[player][1],
                        SHIP_HP, SHIP_VISION_RANGE))
            else:
                if self.currency >= TURRET_COST and num_turrets < TURRET_LIMIT_N:
                    self.currency -= TURRET_COST
                    num_turrets += 1
                    self.add_unit(GameEntity(player, next(id_gen), TYPE_TURRET,
                        BASE_LOCATIONS[player][0], BASE_LOCATIONS[player][1],
                        TURRET_HP, TURRET_VISION_RANGE, f=TURRET_INITIAL_FACING))
    def step_attack(self):
        """
        Attack step.
        Ships and turrets can emit attacks.
        """
        player = self.player
        id_gen = self.id_gen
        # spawn ship attacks
        for sid, tid in self.actions_ship_attack:
            if sid not in self.units:continue
            if tid not in self.units and tid not in self.other_half.units:continue
            sunit = self.units[sid]
            if sunit.vtype != TYPE_SHIP:continue
            if sunit.reload != 0:continue
            if tid in self.units:
                tunit = self.units[tid]
            else:
                tunit = self.other_half.units[tid]
            if sunit.distance(tunit) > SHIP_ATTACK_RANGE:continue
            self.ship_attacks.append((SHIP_ATTACK_DELAY, tid))
            sunit.reload = SHIP_ATTACK_RELOAD
        # spawn turret attacks
        for sid in self.actions_turret_attack:
            if sid not in self.units:continue
            sunit = self.units[sid]
            if sunit.vtype != TYPE_TURRET:continue
            if sunit.reload != 0:continue
            self.turret_attacks.append(GameEntity(player, next(id_gen), TYPE_TURRET_ATTACK,
                sunit.x, sunit.y, 1, 0,
                vx = TURRET_ATTACK_V * math.cos(sunit.f), vy = TURRET_ATTACK_V * math.sin(sunit.f)))
            sunit.reload = TURRET_ATTACK_RELOAD
    def step_pickup(self):
        """
        Pick up and drop step.
        Ships can pick up or drop turrets.
        Ships holding a turret at the end of this step have their speed limited.
        Turrets being held at the end of this step have their position set to match the ship.
        """
        for is_pickup, spec in self.actions_pickup:
            if is_pickup:
                sid, tid = spec
                if sid not in self.units or tid not in self.units:continue
                sunit = self.units[sid]
                tunit = self.units[tid]
                if sunit.vtype != TYPE_SHIP or tunit.vtype != TYPE_TURRET:continue
                if sunit.carrying != 0:continue
                if sunit.distance(tunit) > PICKUP_RANGE:continue
                sunit.carrying = tid
            else:
                sid = spec[0]
                if sid not in self.units:continue
                sunit = self.units[sid]
                if sunit.vtype != TYPE_SHIP:continue
                sunit.carrying = 0
        for sunit in self.units.values():
            if sunit.vtype != TYPE_SHIP or sunit.carrying == 0:continue
            tid = sunit.carrying
            tunit = self.units[tid]
            tunit.x = sunit.x
            tunit.y = sunit.y
            speed = math.hypot(sunit.vx, sunit.vy)
            if speed > SHIP_LIMIT_V_CARRY:
                vmul = SHIP_LIMIT_V_CARRY / speed
                sunit.vx *= vmul
                sunit.vy *= vmul
    def step_movement(self):
        """
        Movement step.
        Ships do movement, turrets do aiming.
        """
        for sid, tvx, tvy in self.actions_move:
            if sid not in self.units:continue
            sunit = self.units[sid]
            if sunit.vtype != TYPE_SHIP:continue
            tspeed = math.hypot(tvx, tvy)
            max_speed = SHIP_LIMIT_V if sunit.carrying == 0 else SHIP_LIMIT_V_CARRY
            if tspeed > max_speed:
                tmul = max_speed / tspeed
                tvx *= tmul
                tvy *= tmul
            ax = tvx - sunit.vx
            ay = tvy - sunit.vy
            accel = math.hypot(ax, ay)
            if accel > SHIP_LIMIT_A:
                amul = SHIP_LIMIT_A / accel
                ax *= amul
                ay *= amul
            sunit.ax = ax
            sunit.ay = ay
        for sid, tf in self.actions_turret_turn:
            if sid not in self.units:continue
            sunit = self.units[sid]
            if sunit.vtype != TYPE_TURRET:continue
            df = tf - sunit.f
            if df > TURRET_LIMIT_T:
                df = TURRET_LIMIT_T
            elif df < -TURRET_LIMIT_T:
                df = -TURRET_LIMIT_T
            sunit.vf = df
        for sunit in self.units.values():
            sunit.movement()
        for sattack in self.turret_attacks:
            sattack.movement()
    def step_damage(self):
        """
        Damage step.
        Ship attacks hit and get absorbed.
        Turret attacks deal damage once this tick.
        """
        sunits = self.units
        ounits = self.other_half.units
        ship_attacks = self.ship_attacks
        masked_turrets = set()
        for unit in itertools.chain(sunits.values(), ounits.values()):
            if unit.vtype != TYPE_SHIP:continue
            masked_turrets.add(unit.carrying)
        for i in range(len(ship_attacks))[::-1]:
            delay, tid = ship_attacks[i]
            if delay != 0:continue
            del ship_attacks[i]
            if tid in masked_turrets:continue
            if tid in sunits:
                tunit = sunits[tid]
            elif tid in ounits:
                tunit = ounits[tid]
            else:
                continue
            tunit.hp -= SHIP_ATTACK_DAMAGE
        for attack in self.turret_attacks:
            for unit in ounits.values():
                if unit.distance(attack) <= TURRET_ATTACK_R:
                    unit.hp -= TURRET_ATTACK_DAMAGE
    def step_remove_dead(self):
        """
        Remove dead units step.
        Also removes turret shots that are too far out of bounds to have any future effect.
        For symmetry reasons, this actually removes the other half's dead units rather than its own.
        """
        ounits = self.other_half.units
        turret_attacks = self.turret_attacks
        for unit in list(ounits.values()):
            if unit.hp <= 0:
                del ounits[unit.vid]
        min_x = min(BASE_LOCATIONS[0][0], BASE_LOCATIONS[1][0])
        max_x = max(BASE_LOCATIONS[0][0], BASE_LOCATIONS[1][0])
        min_y = min(BASE_LOCATIONS[0][1], BASE_LOCATIONS[1][1])
        max_y = max(BASE_LOCATIONS[0][1], BASE_LOCATIONS[1][1])
        for unit in ounits.values():
            min_x = min(min_x, unit.x)
            max_x = max(max_x, unit.x)
            min_y = min(min_y, unit.y)
            max_y = max(max_y, unit.y)
        min_x -= TURRET_ATTACK_PRUNE_MARGIN
        max_x += TURRET_ATTACK_PRUNE_MARGIN
        min_y -= TURRET_ATTACK_PRUNE_MARGIN
        max_y += TURRET_ATTACK_PRUNE_MARGIN
        for i in range(len(turret_attacks))[::-1]:
            attack = turret_attacks[i]
            if attack.x < min_x or attack.x > max_x or attack.y < min_y or attack.y > max_y:
                del turret_attacks[i]
    def step_income(self):
        """
        Income step.
        Base generates passive income.
        Ships generate additional income if near a mine.
        """
        currency = self.currency
        sunits = self.units
        currency += PASSIVE_INCOME
        for unit in sunits.values():
            if unit.vtype != TYPE_SHIP:continue
            can_mine = False
            for mine in MINE_LOCATIONS:
                if unit.distance(mine) <= MINE_RANGE:
                    can_mine = True
                    break
            if can_mine:
                currency += MINE_INCOME
        self.currency = currency
    def step_timers(self):
        """
        Timers step.
        Updates all timers.
        """
        sunits = self.units
        ship_attacks = self.ship_attacks
        self.tick_counter += 1
        for i in range(len(ship_attacks)):
            delay, tid = ship_attacks[i]
            ship_attacks[i] = (delay - 1, tid)
        for unit in sunits.values():
            unit.reload = max(0, unit.reload - 1)
    def __str__(self):
        """
        Get a printable summary of the game state.
        Suitable for debugging purposes.
        """
        lines = []
        lines.append(f'Player {self.player}, tick {self.tick_counter}, resources {self.currency}')
        for unit in self.units.values():
            stype = ('Base', 'Ship', 'Turret')[unit.vtype]
            lines.append(f'* {stype} HP={unit.hp} X={unit.x} Y={unit.y}')
        return '\n'.join(lines)

def generate_ids() -> typing.Generator[int, None, None]:
    """
    Generate a sequence of pseudo-random IDs which are unlikely to collide.
    Random seed is generated once when this function is called.
    """
    r = random.getrandbits(64)
    for i in itertools.count():
        h = hash((i,r,1))
        h ^= hash((i,r,2)) << 32
        h &= 2**63 - 1
        yield h

def run_match(p1_name: str, p2_name: str,
    record_path: typing.Optional[pathlib.Path]) -> MatchResult:
    """
    Run a single match between 2 players.
    """
    all_bots = get_all_bots()
    all_bots_map = {bot.name: bot for bot in all_bots}
    p1_info = all_bots_map[p1_name]
    p2_info = all_bots_map[p2_name]
    # start up the bots
    id_gen = generate_ids()
    halfs = [
        HalfState(0, p1_info, id_gen),
        HalfState(1, p2_info, id_gen)
        ]
    for i in range(2):
        halfs[i].link(halfs[1-i])
    p1_init_err = halfs[0].startup()
    p2_init_err = halfs[1].startup()
    if p1_init_err or p2_init_err:
        for half in halfs:
            half.shutdown()
        outcome = bool(p2_init_err) - bool(p1_init_err)
        reason = p1_init_err or p2_init_err
        return MatchResult(outcome, reason, 0)
    # run the match to completion
    for half in halfs:
        half.set_starting_state()
    phases = [
        HalfState.step_query,
        HalfState.step_production,
        HalfState.step_attack,
        HalfState.step_pickup,
        HalfState.step_movement,
        HalfState.step_damage,
        HalfState.step_remove_dead,
        HalfState.step_income,
        HalfState.step_timers,
        ]
    while halfs[0].tick_counter < TICK_LIMIT and all(half.player_alive() for half in halfs):
        for phase_func in phases:
            p1_err = phase_func(halfs[0])
            p2_err = phase_func(halfs[1])
            if p1_err or p2_err:
                for half in halfs:
                    half.shutdown()
                outcome = bool(p2_err) - bool(p1_err)
                reason = p1_err or p2_err
                return MatchResult(outcome, reason, halfs[0].tick_counter)
    outcome = halfs[0].player_alive() - halfs[1].player_alive()
    reason = 'OK'
    for half in halfs:
        half.shutdown()
    return MatchResult(outcome, reason, halfs[0].tick_counter)

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Runs a single match of ShipTurret.')
    parser.add_argument('--p1', required=True,
        help='Name of the bot to use for player 1. Win/Draw/Loss is from player 1\'s perspective.')
    parser.add_argument('--p2', required=True,
        help='Name of the bot to use for player 2.')
    parser.add_argument('--record-to',
        help='File path to record the match replay to. If not specified, no replay is produced.')
    args = parser.parse_args()
    p1_name = args.p1
    p2_name = args.p2
    record_to = args.record_to
    if record_to is not None:
        try:
            record_to = pathlib.Path(record_to)
        except:
            print('Invalid path: ' + record_to)
            exit(1)
    match_result = run_match(p1_name, p2_name, record_to)
    print(match_result)

if __name__ == '__main__':
    main()
