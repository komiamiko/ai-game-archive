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
TURRET_ATTACK_V = 0.2
TURRET_ATTACK_R = 0.35
TURRET_ATTACK_RELOAD = 10
TURRET_ATTACK_DAMAGE = 3
TURRET_HP = 50
HOME_VISION_RANGE = 2
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
        hp: int, reload: int = 0, vx: float = 0, vy: float = 0, f: float = 0,
        ax: float = 0, ay: float = 0, vf: float = 0, carrying: int = 0):
        self.player = player
        self.vid = vid
        self.vtype = vtype
        self.x = x
        self.y = y
        self.hp = hp
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
    """
    def __init__(self, outcome: int, reason: str):
        self.outcome = outcome
        self.reason = reason
    def __str__(self):
        return self.outcome + '\n' + self.reason

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

def add_unit(vdict: dict[int, GameEntity], unit: GameEntity):
    """
    Add a unit, keyed by its ID.
    """
    vdict[unit.vid] = unit

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
    p1_ret = 0
    for command in p1_info.commands[:-1]:
        p1_sub = subprocess.Popen(command)
        p1_ret = p1_sub.wait()
        if p1_ret:
            break
    p1_sub = None
    if not p1_ret:
        command = p1_info.commands[-1]
        p1_sub = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p2_ret = 0
    for command in p2_info.commands[:-1]:
        p2_sub = subprocess.Popen(command)
        p2_ret = p2_sub.wait()
        if p2_ret:
            break
    p2_sub = None
    if not p2_ret:
        command = p2_info.commands[-1]
        p2_sub = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    if p1_ret or p2_ret:
        outcome = bool(p2_ret) - bool(p1_ret)
        reason = 'INITIALIZATION_FAIL'
        if p1_sub is not None:
            try:
                p1_sub.wait(10)
            except subprocess.TimeoutExpired:
                p1_sub.kill()
        if p2_sub is not None:
            try:
                p2_sub.wait(10)
            except subprocess.TimeoutExpired:
                p2_sub.kill()
        return MatchResult(outcome, reason)
    # run the match to completion
    p1_units = {}
    p2_units = {}
    p1_turret_attacks = []
    p2_turret_attacks = []
    ship_attacks = []
    p1_currency = 0
    p2_currency = 0
    id_gen = generate_ids()
    add_unit(p1_units, GameEntity(0, next(id_gen), TYPE_BASE,
        BASE_LOCATIONS[0][0], BASE_LOCATIONS[0][1], BASE_HP))
    for _ in range(SHIP_STARTING_N):
        add_unit(p1_units, GameEntity(0, next(id_gen), TYPE_SHIP,
            BASE_LOCATIONS[0][0], BASE_LOCATIONS[0][1], SHIP_HP))
    add_unit(p2_units, GameEntity(1, next(id_gen), TYPE_BASE,
        BASE_LOCATIONS[1][0], BASE_LOCATIONS[1][1], BASE_HP))
    for _ in range(SHIP_STARTING_N):
        add_unit(p2_units, GameEntity(1, next(id_gen), TYPE_SHIP,
            BASE_LOCATIONS[1][0], BASE_LOCATIONS[1][1], SHIP_HP))

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