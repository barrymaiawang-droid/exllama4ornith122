import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[1] / 'bundle' / 'ornith_serve.py'
spec = importlib.util.spec_from_file_location('template_serve', path)
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)

class Tok:
    config = SimpleNamespace(directory=os.environ.get('ORNITH_MODEL_DIR', str(Path(__file__).resolve().parent / 'fixtures' / 'native_model')))
    def single_id(self, name): return None
    def hf_render_chat_template(self, *args, **kwargs): raise ModuleNotFoundError('transformers')

TOOL = {'type':'function','function':{'name':'write_file','parameters':{'type':'object','properties':{'path':{'type':'string'}}}}}

class ChatTemplate(unittest.TestCase):
    def setUp(self):
        self.engine=serve.Engine(Tok(),None,SimpleNamespace(),262144,2,'template-test')

    def test_multistep_history_keeps_instructions_tools_and_thinking_control(self):
        messages=[{'role':'system','content':'Build the project.'},
                  {'role':'user','content':'Make a page.'},
                  {'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'write_file','arguments':'{"path":"index.html"}'}}]},
                  {'role':'tool','content':'Written.'},
                  {'role':'developer','content':'Keep it responsive.'},
                  {'role':'system','content':'Continue the task.'},
                  {'role':'user','content':'Add navigation.'}]
        before=json.dumps(messages)
        text=self.engine.render(messages,tools=[TOOL],enable_thinking=False)
        for expected in ['Build the project.','Keep it responsive.','Continue the task.',
                         '<tools>','<function=write_file>','<parameter=path>','<tool_response>']:
            self.assertIn(expected,text)
        self.assertEqual(text.count('<|im_start|>system\n'),1)
        self.assertTrue(text.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        self.assertEqual(json.dumps(messages),before)

    def test_text_parts_remain_text(self):
        text=self.engine.render([{'role':'user','content':[{'type':'text','text':'First'},{'type':'text','text':'Second'}]}])
        self.assertIn('First\nSecond',text)
        self.assertTrue(text.endswith('<|im_start|>assistant\n<think>\n'))

    def test_invalid_history_is_reported_instead_of_silently_retemplated(self):
        cases=[[],[{'role':'system','content':'No user query.'}],
               [{'role':'unknown','content':'x'}],
               [{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:...'}}]}],
               [{'role':'user','content':'x'},{'role':'assistant','tool_calls':[{'function':{'name':'x','arguments':'not JSON'}}]}]]
        for messages in cases:
            with self.subTest(messages=messages),self.assertRaises(serve.RequestError):
                self.engine.render(messages,tools=[TOOL])

if __name__=='__main__': unittest.main()
