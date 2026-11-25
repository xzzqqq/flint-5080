"""CPU-only Hugging Face reference for the chat front end."""

import copy
import hashlib
import json
import sys
from pathlib import Path


def tool_call(name, arguments, call_id="one"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def case(name, messages, **kwargs):
    return {"name": name, "messages": messages, **kwargs}


def message(role, content, **kwargs):
    return {"role": role, "content": content, **kwargs}


def template_cases(reference):
    user = message("user", "Hello")
    cases = [{k: row[k] for k in ("name", "messages", "reasoning_effort")} for row in reference]
    cases.append(case("default_effort", [user]))
    for effort in [None, "minimal", "low", "medium", "high", "xhigh", "max", 0.5, 1e-6]:
        cases.append(case(f"effort_{effort}", [user], reasoning_effort=effort))

    schema = {
        "type": "object",
        "properties": {"城市": {"type": "array", "items": {"type": "string"}}},
    }
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "description": '查找 <x> & "y"', "parameters": schema},
        },
        {"name": "ping"},
    ]
    cases.extend(
        [
            case("system_only", [message("system", "Brief answers.")]),
            case(
                "conversation",
                [
                    message("system", "Be precise."),
                    user,
                    message("assistant", "你好", name="helper", reasoning_content="Think first"),
                    user,
                ],
            ),
            case(
                "parts",
                [
                    message(
                        "user",
                        [
                            {"type": "text", "text": "a"},
                            {"type": "input_text", "text": "b"},
                            "c",
                            {},
                            {"type": "image_url"},
                        ],
                    )
                ],
            ),
            case(
                "parallel_tools",
                [
                    user,
                    message(
                        "assistant",
                        "",
                        tool_calls=[
                            tool_call(
                                "lookup", '{"a":{"z":1,"A":"你好🚀"},"B":["<>&",true,1e-7,1e20]}'
                            ),
                            tool_call("ping", "", "two"),
                        ],
                    ),
                    message("tool", "first", name="lookup"),
                    message("tool", "second", tool_call_id="two"),
                    message("assistant", "Done."),
                ],
                tools=tools,
            ),
            case(
                "named_call",
                [user, message("assistant", None, name="ping", tool_calls=[tool_call("ping", {})])],
            ),
        ]
    )
    return cases


def error_cases():
    user = message("user", "Hello")
    return [
        case("unknown_role", [message("other", "x")]),
        *(
            case(f"invalid_effort_{effort}", [user], reasoning_effort=effort)
            for effort in ("unknown", 1.5, -0.1)
        ),
        case("audio", [message("user", [{"type": "audio"}])]),
        case("assistant_image", [message("assistant", [{"type": "image"}])]),
        case("unknown_part", [message("user", [{"type": "unknown"}])]),
        case("invalid_name", [message("assistant", "", tool_calls=[tool_call(None, {})])]),
        case(
            "name_mismatch", [message("assistant", "", name="a", tool_calls=[tool_call("b", {})])]
        ),
        case("non_object", [message("assistant", "", tool_calls=[tool_call("a", "[1]")])]),
        case("invalid_json", [message("assistant", "", tool_calls=[tool_call("a", "{")])]),
    ]


def hf_inputs(row):
    inputs = copy.deepcopy({k: v for k, v in row.items() if k != "name"})
    for msg in inputs["messages"]:
        for call in msg.get("tool_calls", []):
            fn = call["function"]
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"] or "{}")
    inputs["conversation"] = inputs.pop("messages")
    return inputs


def generate(path, reference_path):
    import transformers
    from jinja2 import TemplateError

    tok = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    raw_template = (path / "chat_template.jinja").read_bytes()
    reference = json.loads(reference_path.read_bytes())["cases"]
    original_ids = {row["name"]: row["input_ids"] for row in reference}
    cases = template_cases(reference)
    for row in cases:
        inputs = hf_inputs(row)
        rendered = tok.apply_chat_template(**inputs, tokenize=False)
        ids = tok.encode(rendered, add_special_tokens=False)
        assert ids == tok.apply_chat_template(**inputs, tokenize=True, return_dict=False), row[
            "name"
        ]
        if row["name"] in original_ids:
            assert ids == original_ids[row["name"]], row["name"]
        row.update(rendered=rendered, input_ids=ids)
    errors = error_cases()
    for row in errors:
        try:
            tok.apply_chat_template(**hf_inputs(row), tokenize=False)
        except (TemplateError, json.JSONDecodeError) as exc:
            row["error"] = str(exc)
        else:
            raise ValueError(f"Expected template error for {row['name']}")

    texts = {
        "ascii": "The quick brown fox jumps.",
        "python": "def f(x):\n\treturn x * 2\n",
        "rust": 'fn main() {\n    println!("hi");\n}\n',
        "chinese": "你好，世界！测试多字节UTF-8",
        "emoji": "🚀✨🔥",
        "special": "<|message_model|><|content_text|>hi<|end_message|>",
        "spaces": "  Well... yes?!  ",
        "empty": "",
    }
    tokenizer_cases = []
    for name, text in texts.items():
        ids = tok.encode(text, add_special_tokens=False)
        tokenizer_cases.append(
            dict(
                name=name,
                text=text,
                token_ids=ids,
                decoded_keep_special=tok.decode(ids, skip_special_tokens=False),
                decoded_skip_special=tok.decode(ids, skip_special_tokens=True),
            )
        )

    vocab = tok.get_vocab()
    config = json.loads((path / "tokenizer_config.json").read_bytes())
    special = {key: vocab[token] for key, token in config["extra_special_tokens"].items()}
    special.update(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    hashes = {
        f"{Path(file).stem}_sha256": hashlib.sha256((path / file).read_bytes()).hexdigest()
        for file in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")
    }
    return dict(
        provenance=dict(
            model_name=path.name, transformers_version=transformers.__version__, **hashes
        ),
        chat_template=raw_template.decode(),
        special_tokens=special,
        template_cases=cases,
        template_error_cases=errors,
        tokenizer_cases=tokenizer_cases,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            generate(Path(sys.argv[1]), Path("workloads/golden_cases.json")),
            ensure_ascii=False,
            indent=2,
        )
    )
