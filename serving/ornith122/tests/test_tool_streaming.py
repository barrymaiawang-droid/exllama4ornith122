"""v4 ToolStreamer vs the one-shot parse_tool_calls: for well-formed calls the streamed
arguments must concatenate to exactly the one-shot arguments string, under any chunking.
Stdlib only, no model: python3 -I tests/test_tool_streaming.py"""
import json, os, random, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bundle"))
import ornith_serve as S

TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "content": {"type": "string"}}}}},
         {"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
            "count": {"type": "integer"}, "opts": {"type": "object"}, "flag": {"type": "boolean"}}}}}]


def corpus(n):
    roots = [os.path.expanduser("~/dsh"), os.path.expanduser("~/talon-next/tools"), os.path.expanduser("~/llmfleet")]
    out = []
    for r in roots:
        for d, _, fs in os.walk(r):
            if "node_modules" in d or "/." in d:
                continue
            for f in fs:
                if f.endswith((".html", ".py", ".js", ".md", ".json", ".sh", ".css")):
                    try:
                        t = open(os.path.join(d, f), encoding="utf-8").read()
                    except Exception:
                        continue
                    if "</parameter>" in t or "</function>" in t or "</tool_call>" in t or "<|im_" in t:
                        continue   # the regex itself mis-parses such values; out of scope
                    if 0 < len(t) < 60000:
                        out.append(t)
    random.shuffle(out)
    return out[:n]


WS = ["", "\n", " ", "\n\n", "  \n", "\t"]


def val(rng, texts):
    k = rng.random()
    if k < 0.5:
        return rng.choice(texts)
    if k < 0.6:
        return '"quoted json string \\n ok"'          # coerce unquotes this
    if k < 0.7:
        return '"""docstring"""\nimport os\n'         # starts like a literal, is not one
    if k < 0.8:
        return '"a" b'
    return rng.choice(["x", "  spaced  ", "", "line1\nline2\n\n", "é ünïcode 🙂 \\ \"q\""])


def make(rng, texts):
    calls, parts = [], [rng.choice(["", "Sure.\n\n", "ok "])]
    for _ in range(rng.choice([1, 1, 1, 2, 3])):
        if rng.random() < 0.6:
            params = [("file_path", "/tmp/" + str(rng.randrange(99)) + ".html"), ("content", val(rng, texts))]
            name = "write"
        else:
            params = [("path", "a.py"), ("old", val(rng, texts)), ("new", val(rng, texts)),
                      ("count", rng.choice(["3", " 7 ", "x"])), ("opts", rng.choice(['{"a": [1, 2]}', "{bad"])),
                      ("flag", rng.choice(["true", "false"]))]
            rng.shuffle(params)
            name = "edit"
        body = "".join(f"<parameter={p}>{rng.choice(WS)}{v}{rng.choice(WS)}</parameter>{rng.choice(WS)}" for p, v in params)
        parts.append(f"<tool_call>{rng.choice(WS)}<function={name}>{rng.choice(WS)}{body}</function>{rng.choice(WS)}</tool_call>")
        parts.append(rng.choice(["", "\n", " between "]))
    text = "".join(parts)
    if rng.random() < 0.2:   # a special marker split across chunks
        text = text.replace("</function>", "<|im_end|></function>", 1)
    return text


def chunks(rng, text):
    i, out = 0, []
    while i < len(text):
        n = rng.choice([1, 1, 2, 3, 5, 8, 20, 200])
        out.append(text[i:i + n]); i += n
    return out


def stream(text, rng):
    st = S.ToolStreamer(TOOLS)
    start = text.find("<tool_call>")
    args, names = {}, {}
    for c in chunks(rng, text[start:]):
        for e in st.feed(c):
            names.setdefault(e["index"], e["function"].get("name"))
            args[e["index"]] = args.get(e["index"], "") + e["function"]["arguments"]
    for e in st.feed("", final=True):
        args[e["index"]] = args.get(e["index"], "") + e["function"]["arguments"]
    return [(names[i], args[i]) for i in sorted(names)], st


def main():
    rng = random.Random(7)
    texts = corpus(400)
    n = int(os.environ.get("N", 3000))
    for t in range(n):
        text = make(rng, texts)
        _, calls = S.parse_tool_calls(S.scrub(text), tools=TOOLS)
        want = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
        got, st = stream(text, rng)
        assert got == want, (t, text[:300], got[:1], want[:1])
        assert not st.truncated
    # truncation: the streamed JSON still parses, and is flagged
    text = make(random.Random(1), texts)
    cut = text.find("<parameter=content>") + 40
    got, st = stream(text[:cut], random.Random(2))
    assert st.truncated and all(isinstance(json.loads(a), dict) for _, a in got)
    # streaming really is incremental: a 30 kB value arrives in many pieces before the call ends
    big = "<tool_call>\n<function=write>\n<parameter=file_path>\n/x\n</parameter>\n<parameter=content>\n" + "<div>row</div>\n" * 2000
    st = S.ToolStreamer(TOOLS)
    sent = sum(len(e["function"]["arguments"]) for i in range(0, len(big), 7) for e in st.feed(big[i:i + 7]))
    assert sent > 29000, sent
    print(f"ok: {n} random calls byte-identical to the one-shot parse; truncation closes; streaming is incremental")


main()
