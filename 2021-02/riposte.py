"""
This game is played on a 5x5 board, from (0, 0) to (4, 4).
Each player has a hero, which start in opposite corners.
Players take their actions simultaneously.
An action consists of:
* movement - 1 unit in a cardinal direction, or stay still
* whether to attack
When player A's hero ends the turn on the same tile as player B's hero,
  and player A's hero is attacking, player B's hero dies.
Exception: a hero that is attacking but not moving is immune to dying.
Be the last one standing. 200 turn limit before it's a draw.
You get 1 point if you win, 0 points otherwise.
Here's what makes it interesting:
* attacking has a cooldown of 3 turns, meaning that after you attack,
    you will be unable to attack for the next 2 turns
* you have no way to tell if your opponent attacked
Tournament is a round robin format, meaning every bot
  gets to play against every other bot.
Matches between the same pair of bots are in a big block,
  so it is possible for a bot to learn another bot's behaviour.
"""

import abc
import typing

class base_bot(abc.ABC):
    @abc.abstractmethod
    def reset(self, new_opponent: bool):
        pass
    @abc.abstractmethod
    def get_bot_name(self) -> str:
        pass
    @abc.abstractmethod
    def get_player_name(self) -> str:
        pass
    @abc.abstractmethod
    def get_action(self,
        my_location: typing.Tuple[int, int],
        opponent_location: typing.Tuple[int, int],
        attack_cooldown: int) -> typing.Tuple[typing.Tuple[int, int], bool]:
        pass

class random_bot(base_bot):
    """
    Acts randomly.
    """
    def reset(self, no):
        import random
        self._random = random.Random()
    def get_bot_name(self):
        return 'random_bot'
    def get_player_name(self):
        return 'examples'
    def get_action(self, sloc, oloc, cd):
        random = self._random
        movement = [(0, 0)]
        if sloc[0] > 0:
            movement.append((-1, 0))
        if sloc[0] < 4:
            movement.append((1, 0))
        if sloc[1] > 0:
            movement.append((0, -1))
        if sloc[1] < 4:
            movement.append((0, 1))
        movement = random.choice(movement)
        should_attack = cd == 0 and random.randint(1, 2) == 1
        return movement, should_attack

class angry_bot(base_bot):
    """
    Chases the enemy and dive attacks when it has an opportunity.
    """
    def reset(self, no):
        pass
    def get_bot_name(self):
        return 'angry_bot'
    def get_player_name(self):
        return 'examples'
    def get_action(self, sloc, oloc, cd):
        if sloc[0] < oloc[0]:
            movement = (1, 0)
        elif sloc[0] > oloc[0]:
            movement = (-1, 0)
        elif sloc[1] < oloc[1]:
            movement = (0, 1)
        elif sloc[1] > oloc[1]:
            movement = (0, -1)
        else:
            movement = (0, 0)
        is_moving = movement != (0, 0)
        is_move_to_opp = (sloc[0] + movement[0], sloc[1] + movement[1]) == oloc
        do_attack = cd == 0 and is_move_to_opp
        return movement, do_attack

class afk_bot(base_bot):
    """
    Pretends to not be paying attention, and then ripostes in the wrong neighbourhood.
    Hard counters angry_bot. Usually counters are not allowed, but the bot is basic
    enough that it seems justified.
    """
    def reset(self, no):
        import random
        self._random = random.Random()
        self.counter_first = True
    def get_bot_name(self):
        return 'afk_bot'
    def get_player_name(self):
        return 'examples'
    def get_action(self, sloc, oloc, cd):
        random = self._random
        dist = abs(sloc[0] - oloc[0]) + abs(sloc[1] - oloc[1])
        movement = (0, 0)
        do_attack = False
        if cd == 0 and dist == 1:
            if self.counter_first or random.randint(1, 2) == 1:
                do_attack = True
            self.counter_first = False
        return movement, do_attack

def make_all_bots() -> typing.List[base_bot]:
    return [
        random_bot(),
        angry_bot(),
        afk_bot(),
        ]

def run_match(abot: base_bot, bbot: base_bot) -> typing.Tuple[int, int]:
    import itertools
    MAX_ROUNDS = 200
    BOARD_MAX = 4
    VALID_MOVES = set(itertools.product([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)],[False, True]))
    VALID_POSITIONS = set(itertools.product(*[range(BOARD_MAX+1)]*2))
    apos = (0, 0)
    bpos = (BOARD_MAX, BOARD_MAX)
    acd = 0
    bcd = 0
    abot.reset(False)
    bbot.reset(False)
    afail = False
    bfail = False
    for _ in range(MAX_ROUNDS):
        acd = max(0, acd - 1)
        bcd = max(0, bcd - 1)
        aaction = abot.get_action(apos, bpos, acd)
        baction = bbot.get_action(
            (BOARD_MAX-bpos[0],BOARD_MAX-bpos[1]),
            (BOARD_MAX-apos[0],BOARD_MAX-apos[1]),
            bcd)
        if aaction not in VALID_MOVES:
            afail = True
        else:
            (adx, ady), atk = aaction
            if atk and acd:
                afail = True
            elif atk:
                acd = 3
            apos = (apos[0] + adx, apos[1] + ady)
            if apos not in VALID_POSITIONS:
                afail = True
            amoved = bool(adx or ady)
        if baction not in VALID_MOVES:
            bfail = True
        else:
            (bdx, bdy), btk = baction
            if btk and bcd:
                bfail = True
            elif btk:
                bcd = 3
            bpos = (bpos[0] - bdx, bpos[1] - bdy)
            if bpos not in VALID_POSITIONS:
                bfail = True
            bmoved = bool(bdx or bdy)
        if afail or bfail:
            return (1 - afail, 1 - bfail)
        if atk and not (btk and not bmoved):
            bfail = True
        if btk and not (atk and not amoved):
            afail = True
        if afail or bfail:
            return (1 - afail, 1 - bfail)
    return (0, 0)

def run_tournament(all_bots: typing.List[base_bot]) -> typing.List[typing.Tuple[int, str, str]]:
    import itertools
    ITERATIONS_SCALE = 1000 # subject to change
    bot_scores = {}
    for abot, bbot in itertools.combinations(all_bots, 2):
        aname = abot.get_bot_name()
        bname = bbot.get_bot_name()
        ascore = bot_scores.get(aname, 0)
        bscore = bot_scores.get(bname, 0)
        abot.reset(True)
        bbot.reset(True)
        for _ in range(ITERATIONS_SCALE):
            da, db = run_match(abot, bbot)
            ascore += da
            bscore += db
        bot_scores[aname] = ascore
        bot_scores[bname] = bscore
    score_table = list((bot_scores[ibot.get_bot_name()], ibot.get_bot_name(), ibot.get_player_name()) for ibot in all_bots)
    score_table.sort(reverse=True)
    return score_table

def print_score_table(score_table: typing.List[typing.Tuple[int, str, str]]):
    print('\n'.join(f'# {tup[0]:>7} | {tup[1]:>20} by {tup[2]:<20}'.strip() for tup in score_table))

def main():
    all_bots = make_all_bots()
    score_table = run_tournament(all_bots)
    print_score_table(score_table)

if __name__ == '__main__':
    main()