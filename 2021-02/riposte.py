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
    def __str__(self):
        return self.get_bot_name()

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

class cautious_bot(base_bot):
    """
    Attack dives from 2 tiles away, then retreats.
    Tries to win or force a draw.
    """
    def reset(self, no):
        import random
        self._random = random.Random()
    def get_bot_name(self):
        return 'cautious_bot'
    def get_player_name(self):
        return 'komiamiko'
    def get_action(self, sloc, oloc, cd):
        random = self._random
        movement_toward = set()
        movement_any = set()
        if sloc[0] > 0:
            movement_any.add((-1, 0))
        if sloc[0] < 4:
            movement_any.add((1, 0))
        if sloc[1] > 0:
            movement_any.add((0, -1))
        if sloc[1] < 4:
            movement_any.add((0, 1))
        if sloc[0] < oloc[0]:
            movement_toward.add((1, 0))
        elif sloc[0] > oloc[0]:
            movement_toward.add((-1, 0))
        if sloc[1] < oloc[1]:
            movement_toward.add((0, 1))
        elif sloc[1] > oloc[1]:
            movement_toward.add((0, -1))
        movement_away = movement_any - movement_toward
        dist = abs(sloc[0] - oloc[0]) + abs(sloc[1] - oloc[1])
        if dist == 2 and cd == 0 and random.randint(1, 2) == 1 or not movement_away:
            movement = random.choice(list(movement_toward))
            do_attack = cd == 0
            return movement, do_attack
        if cd != 0 or dist <= 2:
            do_attack = cd == 0 and random.randint(1, 2) == 1
            movement = (0, 0) if do_attack else random.choice(list(movement_away))
            return movement, do_attack
        movement_toward.add((0, 0))
        movement = random.choice(list(movement_toward))
        do_attack = False
        return movement, do_attack

class syalis_bot(base_bot):
    """
    Keep track of how often the opponent takes certain actions, to predict them.
    Be pessimistic - assume the opponent is intelligent,
      and assume we lost the round unless we definitely won.
    """
    def reset(self, no):
        import random
        self._random = random.Random()
        if no:
            self._attack_counts = [[[2,1] for _ in range(3)] for _ in range(4)] # threat turns, pre dist, did attack
            self._move_counts = [[[1]*3 for _ in range(9)] for _ in range(3)] # cooldown, pre dist, move toward
        else:
            if self._last_seen is not None:
                predist, lsloc, stw, stk = self._last_seen
                threat_key = min(3, self._threat_turns)
                if predist == 2:
                    self._attack_counts[threat_key][2][1] += 1
                elif predist == 1:
                    self._attack_counts[threat_key][1][1] += 1
                else:
                    self._attack_counts[threat_key][0][stw < 0 or not stk] += 1
        self._cdxs = [1, 0, 0]
        self._last_seen = None
        self._threat_turns = 0
    def get_bot_name(self):
        return 'Syalis Hypothesis'
    def get_player_name(self):
        return 'komiamiko'
    def get_action(self, sloc, oloc, cd):
        import itertools
        random = self._random
        USE_GREEDY = True
        def calc_dist(aloc, bloc):
            return abs(aloc[0]-bloc[0])+abs(aloc[1]-bloc[1])
        if self._last_seen is not None:
            # update observation
            predist, lsloc, stw, stk = self._last_seen
            omdist = calc_dist(lsloc, oloc)
            otw = predist - omdist
            threat_key = min(3, self._threat_turns)
            for i in range(3):
                self._move_counts[i][predist][otw+1] += self._cdxs[i]
            if sloc == oloc:
                self._attack_counts[threat_key][predist][0] += 1
            potk = omdist <= 1 and self._attack_counts[threat_key][predist][1] / sum(self._attack_counts[threat_key][predist])
            self._cdxs = [
                self._cdxs[0] * (1 - potk) + self._cdxs[1],
                self._cdxs[2],
                self._cdxs[0] * potk
                ]
        dist = calc_dist(sloc, oloc)
        if dist <= 1:
            self._threat_turns += 1
        else:
            self._threat_turns = 0
        threat_key = min(3, self._threat_turns)
        movement_toward = set()
        movement_any = set()
        if sloc[0] > 0:
            movement_any.add((-1, 0))
        if sloc[0] < 4:
            movement_any.add((1, 0))
        if sloc[1] > 0:
            movement_any.add((0, -1))
        if sloc[1] < 4:
            movement_any.add((0, 1))
        if sloc[0] < oloc[0]:
            movement_toward.add((1, 0))
        elif sloc[0] > oloc[0]:
            movement_toward.add((-1, 0))
        if sloc[1] < oloc[1]:
            movement_toward.add((0, 1))
        elif sloc[1] > oloc[1]:
            movement_toward.add((0, -1))
        movement_away = movement_any - movement_toward
        movement_sets = [movement_away, [(0, 0)], movement_toward]
        saction_rewards = [[None]*2 for _ in range(3)]
        for stw in range(-1, 2):
            movement_set = movement_sets[stw+1]
            for stk in [False, True]:
                if not movement_set or stk and cd != 0:
                    saction_rewards[stw+1][stk] = -1e10
                    continue
                ivalue = 0
                for otw in range(-1, 2):
                    rdist = dist - stw - otw
                    prdist = dist - otw
                    potw = 0
                    for i in range(3):
                        potw += self._move_counts[i][dist][otw+1] / sum(self._move_counts[i][dist]) * self._cdxs[i]
                    pcollide = 0 if rdist != 0 else 1 / len(movement_set)
                    potk = 0 if prdist > 1 else self._attack_counts[threat_key][dist][1] / sum(self._attack_counts[threat_key][dist])
                    ivalue += potw * (pcollide * (stk * (1 - potk) + \
                        (stk and otw == 0) * potk - 1.3 * potk * (not stk or stw != 0)) + \
                        0.15 * potk)
                saction_rewards[stw+1][stk] = ivalue
        if USE_GREEDY:
            stw, stk = max(random.sample(list(itertools.product(range(3),range(2))),6),key=(lambda x:saction_rewards[x[0]][x[1]]))
        else:
            weights = []
            options = []
            for stw in range(3):
                for stk in range(2):
                    weights.append(10**saction_rewards[stw][stk])
                    options.append((stw, stk))
            stw, stk = random.choices(options, weights)[0]
        movement = random.choice(list(movement_sets[stw]))
        do_attack = bool(stk)
        return movement, do_attack

def make_all_bots() -> typing.List[base_bot]:
    return [
        random_bot(),
        angry_bot(),
        afk_bot(),
        cautious_bot(),
        syalis_bot(),
        ]

def run_match(abot: base_bot, bbot: base_bot) -> typing.Tuple[int, int]:
    import itertools
    import traceback
    import warnings
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
        try:
            aaction = abot.get_action(apos, bpos, acd)
        except Exception as exc:
            aaction = exc
            traceback.print_exc()
        try:
            baction = bbot.get_action(
                (BOARD_MAX-bpos[0],BOARD_MAX-bpos[1]),
                (BOARD_MAX-apos[0],BOARD_MAX-apos[1]),
                bcd)
        except Exception as exc:
            baction = exc
            traceback.print_exc()
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
            if afail:
                warnings.warn(f'{abot} returned invalid move: {aaction}')
            if bfail:
                warnings.warn(f'{bbot} returned invalid move: {baction}')
            return (1 - afail, 1 - bfail)
        if apos == bpos and atk and not (btk and not bmoved):
            bfail = True
        if apos == bpos and btk and not (atk and not amoved):
            afail = True
        if afail or bfail:
            return (1 - afail, 1 - bfail)
    return (0, 0)

def run_tournament(all_bots: typing.List[base_bot], verbose: bool = True) -> typing.List[typing.Tuple[int, str, str]]:
    import itertools
    OUTER_ITERATIONS = 10
    ITERATIONS_SCALE = 200 # subject to change
    bot_scores = {}
    for ii in range(OUTER_ITERATIONS):
        if verbose:
            print(f'big iteration {ii+1} / {OUTER_ITERATIONS}')
        for abot, bbot in itertools.combinations(all_bots, 2):
            aname = abot.get_bot_name()
            bname = bbot.get_bot_name()
            ascore = bot_scores.get(aname, 0)
            bscore = bot_scores.get(bname, 0)
            dascore = 0
            dbscore = 0
            abot.reset(True)
            bbot.reset(True)
            for _ in range(ITERATIONS_SCALE):
                da, db = run_match(abot, bbot)
                dascore += da
                dbscore += db
            if verbose and ii == 0:
                print(f'{aname} vs {bname} - {dascore}:{dbscore}')
            bot_scores[aname] = ascore + dascore
            bot_scores[bname] = bscore + dbscore
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