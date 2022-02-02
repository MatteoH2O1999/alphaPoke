# AI trained with simple RL algorithm but with an expert action space:
# 1: fight to kill
# 2: fight with weak move
# 3: power up
# 4: apply status effect
# 5: sacrifice (switch to a pokémon with low health or really weak)
# 6: defensive switch (switch to a pokémon resistant to the enemy's active pokémon)
# 7: offensive switch (switch to a pokémon supereffective against the enemy's active pokémon or really strong)
# 8: fight predict (use a move predicting a switch)
# 9: heal
import random

from typing import List
from poke_env.environment.battle import Battle, AbstractBattle
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from utils import InvalidArgument
from .basic_rl import SimpleRLAgent


class ExpertRLAgent(SimpleRLAgent):

    def _get_battle_to_state_func(self):
        if self.b_format == 'gen8randombattle':
            return _battle_to_state_gen8random
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    def _get_action_to_move_func(self):
        if self.b_format == 'gen8randombattle':
            return _action_to_move_gen8random
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    def _get_action_space_size(self):
        if self.b_format == 'gen8randombattle':
            return 9
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    def _state_headers(self) -> List[str]:
        if self.b_format == 'gen8randombattle':
            return ['Player HP', 'Opponent HP', 'Fainted Pokémons', 'Opponent fainted Pokémons', 'Stat balance',
                    'Type balance', 'Boosts balance', 'Is dynamaxed', 'Forced switch', 'Can apply status',
                    'Can power up', 'Can heal']
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')

    def _action_space_headers(self) -> List[str]:
        if self.b_format == 'gen8randombattle':
            return ['Fight to kill', 'Fight with a weak move', 'Power up', 'Use a move that applies a status effect',
                    'Sacrifice a weak Pokémon', 'Perform a defensive switch', 'Perform an offensive switch',
                    'Use a move predicting a switch', 'Heal']
        else:
            raise InvalidArgument(f'{self.b_format} is not a valid battle format for this RL agent')


def _action_to_move_gen8random(agent: Player, action: int, battle: Battle) -> BattleOrder:
    if action == 0:
        return _fight_to_kill(agent, battle)
    elif action == 1:
        return _fight_weak_move(agent, battle)
    elif action == 2:
        return _power_up(agent, battle)
    elif action == 3:
        return _status_effect(agent, battle)
    elif action == 4:
        return _sac(agent, battle)
    elif action == 5:
        return _defensive_switch(agent, battle)
    elif action == 6:
        return _offensive_switch(agent, battle)
    elif action == 7:
        return _fight_predict(agent, battle)
    elif action == 8:
        return _heal(agent, battle)
    else:
        raise RuntimeError('???')


def _battle_to_state_gen8random(battle: AbstractBattle):
    to_embed = []

    # Player pokémon hp
    player_hp = 2
    if battle.active_pokemon.current_hp_fraction < 0.66:
        player_hp -= 1
    if battle.active_pokemon.current_hp_fraction < 0.33:
        player_hp -= 1
    to_embed.append(player_hp)

    # Opponent pokémon hp
    opponent_hp = 2
    if battle.opponent_active_pokemon.current_hp_fraction < 0.66:
        opponent_hp -= 1
    if battle.opponent_active_pokemon.current_hp_fraction < 0.33:
        opponent_hp -= 1
    to_embed.append(opponent_hp)

    # Fainted pokémons
    fainted_mons = 0
    for mon in battle.team.values():
        if mon.fainted:
            fainted_mons += 1
    to_embed.append(fainted_mons)

    # Opponent fainted pokémons
    opponent_fainted_mons = 0
    for mon in battle.opponent_team.values():
        if mon.fainted:
            opponent_fainted_mons += 1
    to_embed.append(opponent_fainted_mons)

    # Battle balance stats
    player_mon_stats = sum(battle.active_pokemon.base_stats.values())
    opponent_mon_stats = sum(battle.opponent_active_pokemon.base_stats.values())
    diff = player_mon_stats - opponent_mon_stats
    balance = 0
    if diff < -150:
        balance = -1
    elif diff > 150:
        balance = 1
    to_embed.append(balance)

    # Battle balance damage multipliers
    player_mon_types = battle.active_pokemon.types
    opponent_mon_types = battle.opponent_active_pokemon.types
    player_multiplier = 1.0
    opponent_multiplier = 1.0
    for t in player_mon_types:
        player_multiplier *= battle.opponent_active_pokemon.damage_multiplier(t)
    for t in opponent_mon_types:
        opponent_multiplier *= battle.active_pokemon.damage_multiplier(t)
    type_balance = player_multiplier - opponent_multiplier
    if type_balance < 0:
        type_balance = 1
    elif type_balance > 0:
        type_balance = -1
    else:
        type_balance = 0
    to_embed.append(type_balance)

    # Boosts balance
    boost_balance = 0
    boost_balance += sum(battle.active_pokemon.boosts.values())
    boost_balance -= sum(battle.opponent_active_pokemon.boosts.values())
    if boost_balance < 1:
        boost_balance = -1
    elif boost_balance > 1:
        boost_balance = 1
    else:
        boost_balance = 0
    to_embed.append(boost_balance)

    # Is dynamaxed
    is_dyna = 0
    if battle.active_pokemon.is_dynamaxed:
        is_dyna = 1
    to_embed.append(is_dyna)

    # Force switch
    forced_switch = 0
    if battle.force_switch:
        forced_switch = 1
    to_embed.append(forced_switch)

    # Can apply status
    can_status = 0
    for move in battle.available_moves:
        if move.status:
            can_status = 1
    to_embed.append(can_status)

    # Can power up
    can_power_up = 0
    for move in battle.available_moves:
        if move.self_boost:
            if sum(move.self_boost.values()) > 0:
                can_power_up = 1
    to_embed.append(can_power_up)

    # Can heal
    can_heal = 0
    for move in battle.available_moves:
        if move.heal > 0:
            can_heal = 1
    to_embed.append(can_heal)

    return tuple(to_embed)


# 1: fight to kill
def _fight_to_kill(agent: Player, battle: Battle):
    if battle.force_switch:
        return agent.choose_random_move(battle)
    opponent_mon = battle.opponent_active_pokemon
    stats = [sum(mon.stats.values()) for mon in battle.team.values()]
    max_stats = max(stats)
    mon_stats = sum(battle.active_pokemon.stats.values())
    for t in battle.active_pokemon.types:
        mon_stats *= opponent_mon.damage_multiplier(t)
    dyna = False
    mega = False
    z_move = False
    if mon_stats >= max_stats and battle.can_mega_evolve:
        mega = True
    if mon_stats >= max_stats and battle.can_z_move and not mega:
        z_move = True
    if mon_stats >= max_stats and battle.can_dynamax and not mega and not z_move:
        dyna = True
    best_move = None
    best_value = float('-inf')
    for move in battle.available_moves:
        if move.current_pp > 0 and move.base_power > 0:
            move_value = move.base_power * opponent_mon.damage_multiplier(move) * move.accuracy
            if move.type in battle.active_pokemon.types:
                move_value *= 1.5
            if move_value > best_value:
                best_move = move
                best_value = move_value
    if best_move:
        return agent.create_order(best_move, dynamax=dyna, mega=mega, z_move=z_move)
    else:
        if len(battle.available_moves) == 0:
            return agent.choose_random_move(battle)
        random_move = random.randint(0, len(battle.available_moves) - 1)
        return agent.create_order(battle.available_moves[random_move], dynamax=dyna, mega=mega, z_move=z_move)


# 2: fight with weak move
def _fight_weak_move(agent: Player, battle: Battle):
    if battle.force_switch:
        return agent.choose_random_move(battle)
    opponent_mon = battle.opponent_active_pokemon
    best_move = None
    best_value = float('inf')
    for move in battle.available_moves:
        if move.current_pp > 0 and move.base_power > 0:
            move_value = (1 / move.base_power) * opponent_mon.damage_multiplier(move) * move.accuracy
            if move_value > best_value:
                best_move = move
                best_value = move_value
    if best_move:
        return agent.create_order(best_move)
    else:
        if len(battle.available_moves) == 0:
            return agent.choose_random_move(battle)
        random_move = random.randint(0, len(battle.available_moves) - 1)
        return agent.create_order(battle.available_moves[random_move])


# 3: power up
def _power_up(agent: Player, battle: Battle):
    if battle.force_switch:
        return agent.choose_random_move(battle)
    boosting_moves = []
    for move in battle.available_moves:
        if move.self_boost:
            if sum(move.self_boost.values()) > 0:
                boosting_moves.append(move)
    best_value = float('-inf')
    best_move = None
    if len(boosting_moves) > 0:
        for move in boosting_moves:
            move_value = sum(move.self_boost.values())
            if move_value > best_value:
                best_move = move
                best_value = move_value
    if best_move:
        return agent.create_order(best_move)
    else:
        if len(battle.available_moves) == 0:
            return agent.choose_random_move(battle)
        random_move = random.randint(0, len(battle.available_moves) - 1)
        return agent.create_order(battle.available_moves[random_move])


# 4: apply status effect
def _status_effect(agent: Player, battle: Battle):
    if battle.force_switch:
        return agent.choose_random_move(battle)
    status_moves = []
    for move in battle.available_moves:
        if move.status:
            status_moves.append(move)
    if len(status_moves) > 0:
        return agent.create_order(random.choice(status_moves))
    else:
        if len(battle.available_moves) == 0:
            return agent.choose_random_move(battle)
        random_move = random.randint(0, len(battle.available_moves) - 1)
        return agent.create_order(battle.available_moves[random_move])


# 5: sacrifice (switch to a pokémon with low health or really weak)
def _sac(agent: Player, battle: Battle):
    should_switch = True
    best_mon = None
    best_value = float('inf')
    if not battle.force_switch:
        best_mon = battle.active_pokemon
        best_value = sum(battle.active_pokemon.stats.values()) * battle.active_pokemon.current_hp_fraction
        should_switch = False
    for mon in battle.available_switches:
        mon_value = sum(mon.stats.values()) * mon.current_hp_fraction
        if mon_value < best_value:
            best_value = mon_value
            best_mon = mon
            should_switch = True
    if should_switch:
        if best_mon:
            return agent.create_order(best_mon)
        else:
            available_mons = [mon for mon in battle.team.values() if not mon.fainted]
            # This is a bug: sometimes the last pokémon does not get listed as available. Choosing a random move will
            # choose that pokémon
            if len(available_mons) > 0:
                return agent.create_order(random.choice(available_mons))
            else:
                return agent.choose_random_move(battle)
    else:
        opponent_mon = battle.opponent_active_pokemon
        best_move = None
        best_value = float('-inf')
        for move in battle.available_moves:
            if move.current_pp > 0 and move.base_power > 0:
                move_value = move.base_power * opponent_mon.damage_multiplier(move) * move.accuracy
                if move_value > best_value:
                    best_move = move
                    best_value = move_value
        if best_move:
            return agent.create_order(best_move)
        else:
            if len(battle.available_moves) == 0:
                return agent.choose_random_move(battle)
            random_move = random.randint(0, len(battle.available_moves) - 1)
            return agent.create_order(battle.available_moves[random_move])


# 6: defensive switch (switch to a pokémon resistant to the enemy's active pokémon)
def _defensive_switch(agent: Player, battle: Battle):
    known_enemy_moves = list(battle.opponent_active_pokemon.moves.values())
    should_switch = True
    best_mon = None
    best_value = float('inf')
    if not battle.force_switch:
        best_mon = battle.active_pokemon
        multipliers = []
        for t in battle.opponent_active_pokemon.types:
            multipliers.append(_switch_aux(best_mon, t))
        for move in known_enemy_moves:
            multipliers.append(_switch_aux(best_mon, move.type))
        best_value = sum(multipliers)
        should_switch = False
    for mon in battle.available_switches:
        multipliers = []
        for t in battle.opponent_active_pokemon.types:
            multipliers.append(_switch_aux(mon, t))
        for move in known_enemy_moves:
            multipliers.append(_switch_aux(mon, move.type))
        mon_value = sum(multipliers)
        if mon_value < best_value:
            best_value = mon_value
            best_mon = mon
            should_switch = True
    if should_switch:
        if best_mon:
            return agent.create_order(best_mon)
        else:
            available_mons = [mon for mon in battle.team.values() if not mon.fainted]
            # This is a bug: sometimes the last pokémon does not get listed as available. Choosing a random move will
            # choose that pokémon
            if len(available_mons) > 0:
                return agent.create_order(random.choice(available_mons))
            else:
                return agent.choose_random_move(battle)
    else:
        opponent_mon = battle.opponent_active_pokemon
        best_move = None
        best_value = float('-inf')
        for move in battle.available_moves:
            if move.current_pp > 0 and move.base_power > 0:
                move_value = move.base_power * opponent_mon.damage_multiplier(move) * move.accuracy
                if move_value > best_value:
                    best_move = move
                    best_value = move_value
        if best_move:
            return agent.create_order(best_move)
        else:
            if len(battle.available_moves) == 0:
                return agent.choose_random_move(battle)
            random_move = random.randint(0, len(battle.available_moves) - 1)
            return agent.create_order(battle.available_moves[random_move])


# 7: offensive switch (switch to a pokémon supereffective against the enemy's active pokémon or really strong)
def _offensive_switch(agent: Player, battle: Battle):
    opponent_mon = battle.opponent_active_pokemon
    should_switch = True
    best_mon = None
    best_value = float('-inf')
    if not battle.force_switch:
        best_mon = battle.active_pokemon
        stats = sum(best_mon.stats.values())
        for t in best_mon.types:
            stats *= _switch_aux(opponent_mon, t)
        for move in best_mon.moves.values():
            if move.current_pp > 0 and move.base_power > 0:
                stats *= _switch_aux(opponent_mon, move.type)
        best_value = stats
        should_switch = False
    for mon in battle.available_switches:
        stats = sum(mon.stats.values())
        for t in mon.types:
            stats *= _switch_aux(opponent_mon, t)
        for move in mon.moves.values():
            if move.current_pp > 0 and move.base_power > 0:
                stats *= _switch_aux(opponent_mon, move.type)
        mon_value = stats
        if mon_value > best_value:
            best_mon = mon
            best_value = mon_value
            should_switch = True
    if should_switch:
        if best_mon:
            return agent.create_order(best_mon)
        else:
            available_mons = [mon for mon in battle.team.values() if not mon.fainted]
            # This is a bug: sometimes the last pokémon does not get listed as available. Choosing a random move will
            # choose that pokémon
            if len(available_mons) > 0:
                return agent.create_order(random.choice(available_mons))
            else:
                return agent.choose_random_move(battle)
    else:
        opponent_mon = battle.opponent_active_pokemon
        best_move = None
        best_value = float('-inf')
        for move in battle.available_moves:
            if move.current_pp > 0 and move.base_power > 0:
                move_value = move.base_power * opponent_mon.damage_multiplier(move) * move.accuracy
                if move_value > best_value:
                    best_move = move
                    best_value = move_value
        if best_move:
            return agent.create_order(best_move)
        else:
            if len(battle.available_moves) == 0:
                return agent.choose_random_move(battle)
            random_move = random.randint(0, len(battle.available_moves) - 1)
            return agent.create_order(battle.available_moves[random_move])


# 8: fight predict (use a move predicting a switch)
def _fight_predict(agent: Player, battle: Battle):
    if battle.force_switch:
        return agent.choose_random_move(battle)
    best_move = None
    best_value = float('-inf')
    strong_against_types = []
    for t in PokemonType:
        if round(battle.active_pokemon.damage_multiplier(t)) >= 2:
            strong_against_types.append(t)
    for move in battle.available_moves:
        multiplier = 1.0
        if move.current_pp > 0 and move.base_power > 0:
            multiplier *= battle.active_pokemon.damage_multiplier(move.type)
            if multiplier > best_value:
                best_value = multiplier
                best_move = move
    if best_move:
        return agent.create_order(best_move)
    else:
        if len(battle.available_moves) == 0:
            return agent.choose_random_move(battle)
        random_move = random.randint(0, len(battle.available_moves) - 1)
        return agent.create_order(battle.available_moves[random_move])


# 9: heal
def _heal(agent: Player, battle: Battle):
    if battle.force_switch:
        return agent.choose_random_move(battle)
    best_move = None
    best_value = float('-inf')
    for move in battle.available_moves:
        if move.heal > best_value:
            best_value = move.heal
            best_move = move
    if best_move:
        return agent.create_order(best_move)
    else:
        if len(battle.available_moves) == 0:
            return agent.choose_random_move(battle)
        random_move = random.randint(0, len(battle.available_moves) - 1)
        return agent.create_order(battle.available_moves[random_move])


def _switch_aux(mon, t):
    return mon.damage_multiplier(t)**3


def _boosts_aux(battle, boost):
    boost_balance = 0
    boost_balance += battle.active_pokemon.boosts[boost]
    boost_balance -= battle.opponent_active_pokemon.boosts[boost]
    if boost_balance < 0:
        boost_balance = -1
    elif boost_balance > 0:
        boost_balance = 1
    else:
        boost_balance = 0
    return boost_balance
    
