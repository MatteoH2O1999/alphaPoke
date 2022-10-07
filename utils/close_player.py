import asyncio

from logging import disable, NOTSET, CRITICAL
from threading import Thread

from poke_env.player.player import Player


def close_player(player: Player):
    t = _ClosePlayerThread(player)
    t.start()
    t.join()


class _ClosePlayerThread(Thread):
    def __init__(self, player: Player, *args, **kwargs):
        self.player = player
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.stop_player())

    async def stop_player(self):
        from agents.base_classes.tf_player import TFPlayer

        disable(CRITICAL)
        await self.player.stop_listening()
        if isinstance(self.player, TFPlayer):
            self.clean_tf_player()
        disable(NOTSET)

    def clean_tf_player(self):
        from agents.base_classes.tf_player import TFPlayer

        assert isinstance(self.player, TFPlayer)
        self.player.policy = None
