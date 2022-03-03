from utils.action_space_init import init_action_space


def test_init_actin_space():
    assert init_action_space(0) == []
    for i in range(1, 50):
        assert init_action_space(i) == [0 for i in range(i)]
