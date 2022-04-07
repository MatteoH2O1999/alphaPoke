from unittest.mock import MagicMock, PropertyMock, patch

from utils.get_smogon_data import get_random_battle_learnset


def test_get_random_battle_learnset_gen8():
    with patch("requests.get") as mock_req_get, patch("json.loads") as mock_loads:
        data = MagicMock()
        mock_req_get.return_value = data
        dummy_dict = {"pikachu": {}, "Mimikyu": {}, "Rattata": {}}
        mock_loads.return_value = dummy_dict
        learnset = get_random_battle_learnset(8)
        mock_req_get.assert_called_once_with(
            "https://raw.githubusercontent.com/pkmn/randbats/main/data/gen8randombattle.json"
        )
        mock_loads.assert_called_once_with(data.content)
        assert "pikachukalos" in learnset.keys()
        assert "mimikyubusted" in learnset.keys()
        assert "rattata" in learnset.keys()
        assert "eiscuenoice" not in learnset.keys()


def test_get_random_battle_learnset_gen6():
    with patch("requests.get") as mock_req_get, patch("json.loads") as mock_loads:
        data = MagicMock()
        mock_req_get.return_value = data
        dummy_dict = {"Gastrodon": {}, "Zygarde": {"items": ["item"]}, "Eiscue": {}}
        mock_loads.return_value = dummy_dict
        learnset = get_random_battle_learnset(6)
        mock_req_get.assert_called_once_with(
            "https://raw.githubusercontent.com/pkmn/randbats/main/data/gen6randombattle.json"
        )
        mock_loads.assert_called_once_with(data.content)
        assert "gastrodoneast" in learnset.keys()
        assert "eiscuenoice" in learnset.keys()
        assert "zygardecomplete" in learnset.keys()
        assert "pikachupartner" not in learnset.keys()
        assert learnset["gastrodoneast"]["items"] == []
        assert learnset["zygarde"]["items"] == ["item"]
        assert learnset["zygardecomplete"]["items"] == ["item"]
