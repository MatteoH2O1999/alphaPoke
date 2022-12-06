#
# A pokémon showdown battle-bot project based on reinforcement learning techniques.
# Copyright (C) 2022 Matteo Dell'Acqua
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import pytest

from logging import CRITICAL, NOTSET
from unittest.mock import MagicMock, call, patch

from utils.close_player import close_player, _ClosePlayerThread


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


def test_close_player_function():
    with patch("utils.close_player._ClosePlayerThread") as mock_thread:
        mock_player = MagicMock()
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        close_player(mock_player)

        mock_thread.assert_called_once_with(mock_player)
        mock_thread_instance.start.assert_called_once()
        mock_thread_instance.join.assert_called_once_with()
        mock_player.stop_listening.assert_not_called()


def test_close_player_thread_init():
    p = MagicMock()
    t = _ClosePlayerThread(p)
    assert t.player is p


@pytest.mark.asyncio
async def test_close_player_thread_test_stop_player():
    with patch("utils.close_player.disable") as mock_disable:
        p = MagicMock()
        p.stop_listening = AsyncMock()
        t = _ClosePlayerThread(p)

        await t.stop_player()

        mock_disable.assert_has_calls([call(CRITICAL), call(NOTSET)])
        p.stop_listening.assert_called_once()


def test_close_player_thread_run():
    with patch("asyncio.new_event_loop") as mock_new_loop:
        mock_loop = MagicMock()
        mock_new_loop.return_value = mock_loop

        p = MagicMock()
        t = _ClosePlayerThread(p)
        t.stop_player = MagicMock()
        mock_coro = MagicMock()
        t.stop_player.return_value = mock_coro

        t.run()

        mock_new_loop.assert_called_once()
        mock_loop.run_until_complete.assert_called_once_with(mock_coro)
