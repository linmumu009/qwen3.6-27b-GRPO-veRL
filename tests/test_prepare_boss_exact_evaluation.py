import json

from scripts.prepare_boss_exact_evaluation import qwen_output_to_openai_messages


def test_qwen_output_to_openai_messages_preserves_commands_and_terminal_answer():
    output = """先查看数据库。
</think>

<tool_call>
<function=bash>
<parameter=command>
sqlite3 /workspace/logistics.sqlite \"SELECT a, SUM(b) FROM fact_cost GROUP BY a\"
</parameter>
<parameter=timeout>
30
</parameter>
</function>
</tool_call>
user
<tool_response>
甲|12.5
乙|9.0
</tool_response>
assistant
<think>
结果已足够。
</think>

最终结果：甲 12.5，乙 9.0。"""

    messages, audit = qwen_output_to_openai_messages(output)

    assert [message["role"] for message in messages] == ["assistant", "tool", "assistant"]
    call = messages[0]["tool_calls"][0]
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {
        "command": 'sqlite3 /workspace/logistics.sqlite "SELECT a, SUM(b) FROM fact_cost GROUP BY a"',
        "timeout": "30",
    }
    assert messages[1]["tool_call_id"] == call["id"]
    assert messages[-1]["content"].endswith("最终结果：甲 12.5，乙 9.0。")
    assert audit == {
        "tool_calls": 1,
        "tool_responses": 1,
        "missing_tool_responses": 0,
        "terminal_assistant": True,
    }


def test_qwen_output_to_openai_messages_preserves_incomplete_terminal_tool_call():
    output = """<tool_call>
<function=read>
<parameter=path>
/workspace/schema_dictionary.md
</parameter>
</function>
</tool_call>
user
<tool_response>
schema
</tool_response>
assistant
"""

    messages, audit = qwen_output_to_openai_messages(output)

    assert [message["role"] for message in messages] == ["assistant", "tool"]
    assert audit["terminal_assistant"] is False


def test_qwen_output_to_openai_messages_preserves_unanswered_terminal_call():
    messages, audit = qwen_output_to_openai_messages(
        "<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>"
    )

    assert len(messages) == 1
    assert messages[0]["tool_calls"][0]["function"]["name"] == "bash"
    assert audit["missing_tool_responses"] == 1
    assert audit["terminal_assistant"] is False


def test_qwen_output_to_openai_messages_preserves_token_truncated_terminal_call():
    messages, audit = qwen_output_to_openai_messages(
        "分析已完成。\n<tool_call>\n<function=bash>\n<parameter=command>\n"
        'sqlite3 /workspace/logistics.sqlite "SELECT AVG(value)'
    )

    assert len(messages) == 1
    assert messages[0]["content"] == "分析已完成。"
    call = messages[0]["tool_calls"][0]
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {
        "command": 'sqlite3 /workspace/logistics.sqlite "SELECT AVG(value)'
    }
    assert audit["tool_calls"] == 1
    assert audit["missing_tool_responses"] == 1
    assert audit["truncated_terminal_tool_calls"] == 1
    assert audit["terminal_assistant"] is False


def test_qwen_output_to_openai_messages_preserves_parallel_tool_group():
    output = """并行检查两个文件。
<tool_call>
<function=read><parameter=path>/workspace/a.md</parameter></function>
</tool_call>
<tool_call>
<function=read><parameter=path>/workspace/b.md</parameter></function>
</tool_call>
user
<tool_response>
A
</tool_response>
<tool_response>
B
</tool_response>
assistant
结论完整。"""

    messages, audit = qwen_output_to_openai_messages(output)

    assert [message["role"] for message in messages] == [
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert len(messages[0]["tool_calls"]) == 2
    assert messages[1]["content"] == "A"
    assert messages[2]["content"] == "B"
    assert audit["tool_calls"] == 2


def test_qwen_output_to_openai_messages_audits_missing_parallel_response():
    output = """<tool_call>
<function=read><parameter=path>/workspace/a.md</parameter></function>
</tool_call>
<tool_call>
<function=read><parameter=path>/workspace/b.md</parameter></function>
</tool_call>
user
<tool_response>A</tool_response>
assistant
最终结论。"""

    messages, audit = qwen_output_to_openai_messages(output)

    assert len(messages[0]["tool_calls"]) == 2
    assert [message["role"] for message in messages] == ["assistant", "tool", "assistant"]
    assert audit["missing_tool_responses"] == 1
