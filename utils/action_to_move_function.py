# Return a function with the correct battle format from AI action space to showdown env.
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder


def action_to_move_gen8random(agent, action: int, battle: Battle) -> BattleOrder:
    if action == -1:
        return ForfeitBattleOrder()
    elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
    ):
        return agent.create_order(battle.available_moves[action])
    elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0
            <= action - 4
            < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
    ):
        return agent.create_order(
            battle.active_pokemon.available_z_moves[action - 4], z_move=True
        )
    elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
    ):
        return agent.create_order(battle.available_moves[action - 8], mega=True)
    elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
    ):
        return agent.create_order(battle.available_moves[action - 12], dynamax=True)
    elif 0 <= action - 16 < len(battle.available_switches):
        return agent.create_order(battle.available_switches[action - 16])
    else:
        return agent.choose_random_move(battle)
