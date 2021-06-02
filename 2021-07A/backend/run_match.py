"""
Runs a single match between 2 bots, reports the result, then exits.
"""

import argparse
import itertools
import math
import pathlib
import subprocess
import typing
from .bot_info import BotInfo, get_all_bots

# --- game related constants ---

BASE_LOCATIONS = [(-10,0),(10,0)]
MINE_LOCATIONS = [(0,-4),(0,1),(-2,6),(2,6),(-6,10),(6,10)]
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
TURRET_HP = 50
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
        self.f = facing
        self.ax = ax
        self.ay = ay
        self.vf = vf
        self.carrying = carrying
    def movement(self):
        self.vx += ax
        self.vy += ay
        self.ax = 0
        self.ay = 0
        self.x += vx
        self.y += vy
        self.f += vf
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
    * OK a player's home base is destroyed
    * INITIALIZATION_FAIL a bot failed during initialization
    * TIMEOUT a bot exceeded the hard time limit per tick
    * DIED_EARLY a bot process died before the match was over
    * INVALID_ACTION a bot issued an invalid instruction
    """
    def __init__(self, outcome: int, reason: str):
        self.outcome = outcome
        self.reason = reason
    def __str__(self):
        return self.outcome + '\n' + self.reason

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
            text=True)
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
                    bot_sub.stdin.write(input='end_game\n')
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
        for unit in sunits:
            if unit.vtype == TYPE_BASE:
                visible_bases[unit.vid] = unit
            elif unit.vtype == TYPE_SHIP:
                visible_ships[unit.vid] = unit
            elif unit.vtype == TYPE_TURRET:
                visible_turrets[unit.vid] = unit
        for ounit in ounits:
            seen = False
            for unit in sunits:
                if unit.distance(ounit) <= unit.vision:
                    seen = True
                    break
            if seen:
                if unit.vtype == TYPE_BASE:
                    visible_bases[unit.vid] = unit
                elif unit.vtype == TYPE_SHIP:
                    visible_ships[unit.vid] = unit
                elif unit.vtype == TYPE_TURRET:
                    visible_turrets[unit.vid] = unit
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
        for unit in visible_bases:
            lines.append(f'home {unit.vid} {fix_player(unit.player)} {unit.x} {unit.y} {unit.hp}')
        for unit in visible_ships:
            lines.append(f'ship {unit.vid} {fix_player(unit.player)} {unit.x} {unit.y} {unit.hp}' + \
                f' {unit.vx} {unit.vy} {unit.reload} {unit.carrying}')
        for unit in visible_turrets:
            lines.append(f'turret {unit.vid} {fix_player(unit.player)} {unit.x} {unit.y} {unit.hp}' + \
                f' {unit.f} {unit.reload}')
        for delay, target_id in visible_ship_attacks:
            lines.append(f'ship_attack {target_id} {delay}')
        for unit in visible_turret_attacks:
            lines.append(f'turret_attack {unit.player} {unit.x} {unit.y} {unit.vx} {unit.vy}')
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
        pass
    def step_attack(self):
        pass
    def step_pickup(self):
        pass
    def step_movement(self):
        pass
    def step_damage(self):
        pass
    def step_remove_dead(self):
        pass
    def step_income(self):
        pass
    def step_timers(self):
        pass

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
        return h

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
        return MatchResult(outcome, reason)
    # run the match to completion
    for half in halfs:
        half.set_starting_state()

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