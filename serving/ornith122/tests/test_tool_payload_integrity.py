"""Native Qwen XML string arguments must retain the exact file/edit payload."""
import json
import random
import unittest
from pathlib import Path
import test_repetition_guard_http as harness

TOOLS=[{'type':'function','function':{'name':'write','parameters':{'type':'object','properties':{
    'file_path':{'type':'string'},'content':{'type':'string'}}}}}]

def wire(payload):
    return '<tool_call>\n<function=write>\n<parameter=file_path>\n/x\n</parameter>\n<parameter=content>\n'+payload+'\n</parameter>\n</function>\n</tool_call>'

class ToolPayloadIntegrity(unittest.TestCase):
    def test_strings_roundtrip_through_both_response_paths(self):
        mod=harness.load_module()
        fixtures=['    return value\n','\n\n  indented and padded  \n\n','"literal JSON string"',
                  'transform: translate(-50%, -50%) rotateZ(-0deg);\n',
                  'const x = `hello ${name}`;\n', 'é 🙂 \\ "quoted"\r\n',
                  'const marker = "<|im_end|>";\nconst thought = "<think>literal</think>";\n']
        fixtures.extend(p.read_text() for p in sorted((Path(__file__).parent/'fixtures').glob('*.html')))
        for index,payload in enumerate(fixtures):
            with self.subTest(fixture=index):
                text=wire(payload)
                _,calls=mod.parse_tool_calls(text,tools=TOOLS)
                self.assertEqual(json.loads(calls[0]['function']['arguments'])['content'],payload)
                for seed in range(3):
                    rng=random.Random(seed);streamer=mod.ToolStreamer(TOOLS);parts=[];at=0
                    while at<len(text):
                        n=rng.randrange(1,65)
                        parts.extend(streamer.feed(text[at:at+n]));at+=n
                    parts.extend(streamer.feed('',final=True))
                    args=''.join(p['function']['arguments'] for p in parts)
                    self.assertEqual(json.loads(args),{'file_path':'/x','content':payload})

if __name__=='__main__': unittest.main()
