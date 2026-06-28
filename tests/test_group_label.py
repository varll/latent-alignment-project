from latent_alignment.group_label import (
    DEFAULT_CATEGORIES,
    EXTENDED_CATEGORIES,
    EXTENDED_EXTRA_GROUPS,
    OTHER_CATEGORY,
    TOXIGEN_TARGET_GROUPS,
    build_group_messages,
    parse_group_label,
)

CATS = list(DEFAULT_CATEGORIES)


def test_default_categories_are_toxigen_groups_plus_other() -> None:
    assert DEFAULT_CATEGORIES[-1] == OTHER_CATEGORY
    assert tuple(DEFAULT_CATEGORIES[:-1]) == TOXIGEN_TARGET_GROUPS
    assert "women" in TOXIGEN_TARGET_GROUPS
    assert "muslim folks" in TOXIGEN_TARGET_GROUPS
    # The taxonomy carries the two ToxiGen black/african-american spellings verbatim.
    assert "black/african-american folks" in TOXIGEN_TARGET_GROUPS
    assert "black folks / african-americans" in TOXIGEN_TARGET_GROUPS


def test_build_group_messages_lists_categories_and_statement() -> None:
    messages = build_group_messages("  Women are great.  ", CATS)
    assert messages[0]["role"] == "system"
    assert "muslim folks" in messages[0]["content"]
    assert OTHER_CATEGORY in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Women are great." in messages[1]["content"]


def test_parse_group_label_prefers_valid_json() -> None:
    out = parse_group_label('{"group": "women", "reason": "about women"}', CATS)
    assert out["group"] == "women"
    assert out["reason"] == "about women"


def test_parse_group_label_is_case_insensitive_and_canonicalizes() -> None:
    out = parse_group_label('{"group": "MUSLIM FOLKS", "reason": "x"}', CATS)
    assert out["group"] == "muslim folks"


def test_parse_group_label_tolerates_prose_around_json() -> None:
    out = parse_group_label('Sure!\n{"group": "jewish folks", "reason": "y"}\nthanks', CATS)
    assert out["group"] == "jewish folks"


def test_parse_group_label_unknown_group_falls_back_to_other() -> None:
    out = parse_group_label('{"group": "martians", "reason": "z"}', CATS)
    assert out["group"] == OTHER_CATEGORY


def test_parse_group_label_keyword_fallback_when_no_json() -> None:
    out = parse_group_label("This clearly targets women in the workplace.", CATS)
    assert out["group"] == "women"


def test_parse_group_label_empty_and_garbage_to_other() -> None:
    assert parse_group_label("", CATS)["group"] == OTHER_CATEGORY
    assert parse_group_label("!!! ??? ...", CATS)["group"] == OTHER_CATEGORY
    assert parse_group_label("{not valid json", CATS)["group"] == OTHER_CATEGORY


def test_parse_group_label_ambiguous_keyword_echo_falls_back_to_other() -> None:
    # A reasoning model that restates the option list must NOT anchor on one long category name
    # (this previously mislabelled 89 unrelated statements as "folks with physical disabilities").
    echo = (
        "Options: asian folks, folks with mental disabilities, "
        "folks with physical disabilities, women. The statement is unclear."
    )
    assert parse_group_label(echo, CATS)["group"] == OTHER_CATEGORY


def test_extended_categories_extend_toxigen_without_breaking_default() -> None:
    assert tuple(DEFAULT_CATEGORIES[:-1]) == TOXIGEN_TARGET_GROUPS  # default unchanged
    assert EXTENDED_CATEGORIES[-1] == OTHER_CATEGORY
    assert tuple(EXTENDED_CATEGORIES[: len(TOXIGEN_TARGET_GROUPS)]) == TOXIGEN_TARGET_GROUPS
    for extra in EXTENDED_EXTRA_GROUPS:
        assert extra in EXTENDED_CATEGORIES
    assert any("immigrant" in c for c in EXTENDED_CATEGORIES)


def test_parse_group_label_respects_custom_categories() -> None:
    cats = ["cats", "dogs", OTHER_CATEGORY]
    assert parse_group_label('{"group": "dogs"}', cats)["group"] == "dogs"
    assert parse_group_label('{"group": "women"}', cats)["group"] == OTHER_CATEGORY
