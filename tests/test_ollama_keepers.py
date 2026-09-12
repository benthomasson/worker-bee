"""Tests for _OllamaProvider keeper-message sprinkling.

Ollama requires at least one user message inside the context window and
evicts from the OLDEST end first. Tool-result loops produce histories with
no plain user text, so _OllamaProvider sprinkles blank "keeper" user
messages throughout the converted list. The meaningful invariant we test:
the gap between consecutive user messages stays bounded (so a user message
is always within reach of the eviction window), and keepers are placed so
they never split an assistant(tool_calls) from its tool results.
"""

from worker_bee.llm import _OllamaProvider


def _make_provider():
    return _OllamaProvider(model="test", base_url="http://localhost:11434")


def _user_text(text="hello"):
    return {"role": "user", "content": text}


def _assistant_tool(tool_name="read_file", tool_input=None):
    """Simulate a converted assistant message with tool_calls."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": tool_name, "arguments": tool_input or {}}}],
    }


def _tool_result(content="ok"):
    return {"role": "tool", "content": content}


def _keepers(converted):
    return [m for m in converted if m["role"] == "user" and m["content"] == " "]


def _max_non_user_gap(converted):
    """Longest run of non-user messages between two user messages (system
    excluded). Bounded gap = a user message is always within the eviction
    window."""
    max_gap = 0
    gap = 0
    for msg in converted[1:]:  # skip system
        if msg["role"] == "user":
            gap = 0
        else:
            gap += 1
            max_gap = max(max_gap, gap)
    return max_gap


def _pairing_intact(converted):
    """No user/keeper message may sit between an assistant(tool_calls) and
    the tool results that follow it. (A keeper is only ever placed before
    an assistant, so the message after a tool_calls assistant must be a tool
    or the end of the list.)"""
    for idx, msg in enumerate(converted):
        if msg["role"] == "assistant" and "tool_calls" in msg:
            if idx + 1 < len(converted):
                assert converted[idx + 1]["role"] == "tool", (
                    f"assistant(tool_calls) at {idx} not followed by a tool msg: "
                    f"got {converted[idx + 1]['role']}"
                )


def _editor_msgs(n_pairs, seed="Begin."):
    """Build a converted-style list the way the editor loop produces one:
    system, user(seed), then n_pairs of assistant(tool_calls) + tool."""
    msgs = [{"role": "system", "content": "sys"}, _user_text(seed)]
    for i in range(n_pairs):
        msgs.append(_assistant_tool(f"tool_{i}"))
        msgs.append(_tool_result(f"res_{i}"))
    return msgs


class TestKeeperSprinkling:
    def test_no_keepers_for_short_history(self):
        prov = _make_provider()
        converted = _editor_msgs(3)
        result = prov._sprinkle_keepers(converted)
        assert _keepers(result) == []
        assert len(result) == len(converted)

    def test_keepers_appear_for_long_history(self):
        prov = _make_provider()
        result = prov._sprinkle_keepers(_editor_msgs(30))
        assert len(_keepers(result)) >= 1
        # The real invariant: bounded gap between user messages.
        assert _max_non_user_gap(result) <= 2 * prov.KEEPER_MAX_GAP

    def test_gap_bounded_for_very_long_history(self):
        prov = _make_provider()
        for n in (40, 100, 250):
            result = prov._sprinkle_keepers(_editor_msgs(n))
            assert _max_non_user_gap(result) <= 2 * prov.KEEPER_MAX_GAP, (
                f"unbounded user gap at n={n}: {_max_non_user_gap(result)}"
            )
            _pairing_intact(result)

    def test_keepers_only_before_assistant(self):
        prov = _make_provider()
        result = prov._sprinkle_keepers(_editor_msgs(30))
        for idx, msg in enumerate(result):
            if msg["role"] == "user" and msg["content"] == " ":
                assert idx + 1 < len(result), "keeper is the last message"
                assert result[idx + 1]["role"] == "assistant", (
                    f"keeper at {idx} not followed by assistant: "
                    f"got {result[idx + 1]['role']}"
                )

    def test_keepers_do_not_split_tool_pairing(self):
        prov = _make_provider()
        result = prov._sprinkle_keepers(_editor_msgs(30))
        _pairing_intact(result)

    def test_real_user_message_resets_gap(self):
        """A genuine user message mid-history acts as a fresh gap anchor, so
        keepers are placed relative to it rather than being crowded to the
        front of the list."""
        prov = _make_provider()
        # Long run, then a real user message, then another long run.
        part1 = _editor_msgs(30)
        part2 = _editor_msgs(30, seed="continue")
        msgs = part1[1:] + [part2[1]] + part2[2:]  # share one system msg
        msgs[0] = {"role": "system", "content": "sys"}
        result = prov._sprinkle_keepers(msgs)
        # Keepers should be distributed on BOTH sides of the real user msg.
        user_idx = [i for i, m in enumerate(result)
                    if m["role"] == "user" and m["content"] == "continue"][0]
        assert _keepers(result[:user_idx]), "no keepers before the real user msg"
        assert _keepers(result[user_idx + 1:]), "no keepers after the real user msg"

    def test_system_message_stays_first(self):
        prov = _make_provider()
        result = prov._sprinkle_keepers(_editor_msgs(30))
        assert result[0] == {"role": "system", "content": "sys"}

    def test_last_message_unchanged(self):
        prov = _make_provider()
        base = _editor_msgs(30)
        result = prov._sprinkle_keepers(base)
        assert result[-1] == base[-1]


class TestConvertMessagesIntegration:
    def test_editor_loop_history_gets_keepers(self):
        """Full _convert_messages path on a realistic editor-loop history."""
        prov = _make_provider()
        messages = [{"role": "user", "content": "Begin."}]
        for i in range(30):
            messages.append({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"tu_{i}",
                             "name": "read_file", "input": {"path": f"f_{i}.py"}}],
            })
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"tu_{i}",
                             "content": f"line {i}"}],
            })
        converted = prov._convert_messages(messages, "You are a helpful assistant.")
        assert converted[0]["role"] == "system"
        assert len(_keepers(converted)) >= 1
        assert _max_non_user_gap(converted) <= 2 * prov.KEEPER_MAX_GAP
        _pairing_intact(converted)

    def test_short_history_passes_through(self):
        prov = _make_provider()
        messages = [
            {"role": "user", "content": "Begin."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_0", "name": "read_file",
                 "input": {"path": "a.py"}}], },
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_0", "content": "print('hi')"}], },
            {"role": "assistant", "content": "Done."},
        ]
        converted = prov._convert_messages(messages, "sys")
        assert _keepers(converted) == []

    def test_only_ollama_provider_uses_keepers(self):
        """The keeper logic must live on _OllamaProvider, not leak to others."""
        import worker_bee.llm as llm
        assert hasattr(llm._OllamaProvider, "_sprinkle_keepers")
        assert not hasattr(llm._OpenAIProvider, "_sprinkle_keepers")
        assert not hasattr(llm._AnthropicProvider, "_sprinkle_keepers")
