"""HTTP regression tests for generation controls reaching the actual Job seam."""
import json
import unittest
import threading
from http.server import ThreadingHTTPServer
import test_repetition_guard_http as harness

class QualityBackend(harness.FakeBackend):
    @staticmethod
    def make_sampler(**kwargs):
        return kwargs

class GenerationContract(unittest.TestCase):
    def setUp(self):
        self.mod = harness.load_module()
        self.engine = self.mod.Engine(harness.FakeTok(), harness.FakeGen(harness.events(['READY'])),
                                     QualityBackend(), 4096, 2, 'contract')
        self.engine.sampling_defaults = {'temperature': .6, 'top_p': .95, 'top_k': 20}
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), self.mod.make_handler(self.engine))
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        harness.JOBS.clear()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, **kwargs):
        return harness.post(self.port, {'messages': [{'role':'user','content':'Reply READY.'}], **kwargs})

    def test_model_defaults_reach_sampler(self):
        status, body = self.request()
        self.assertEqual(status, 200, body)
        self.assertEqual(harness.JOBS[-1]['sampler']['temperature'], .6)
        self.assertEqual(harness.JOBS[-1]['sampler']['top_p'], .95)
        self.assertEqual(harness.JOBS[-1]['sampler']['top_k'], 20)

    def test_openai_tool_definitions_reach_the_native_template_unchanged(self):
        captured=[]
        class RecordingTok(harness.FakeTok):
            def hf_render_chat_template(self,messages,**kwargs):
                captured.append(kwargs.get('tools'))
                return super().hf_render_chat_template(messages,**kwargs)
        self.engine.tok=RecordingTok()
        tools=[{'type':'function','function':{'name':'write','parameters':{'type':'object','properties':{'content':{'type':'string'}}}}}]
        status,body=self.request(tools=tools)
        self.assertEqual(status,200,body)
        self.assertEqual(captured[-1],tools)

    def test_fleet_thinking_default_and_per_request_override(self):
        seen=[]
        class RecordingTok(harness.FakeTok):
            def hf_render_chat_template(self,messages,**kwargs):
                seen.append(kwargs.get('enable_thinking'))
                return super().hf_render_chat_template(messages,**kwargs)
        self.engine.tok=RecordingTok();self.engine.thinking_default=False
        status,body=self.request()
        self.assertEqual(status,200,body);self.assertIs(seen[-1],False)
        self.engine.gen=harness.FakeGen(harness.events(['READY']))
        status,body=self.request(enable_thinking=True)
        self.assertEqual(status,200,body);self.assertIs(seen[-1],True)

    def test_fleet_budget_is_the_default_and_explicit_request_can_override(self):
        self.engine.default_max_tokens=10
        status,body=self.request()
        self.assertEqual(status,200,body);self.assertEqual(harness.JOBS[-1]['max_new_tokens'],10)
        self.engine.gen=harness.FakeGen(harness.events(['READY']))
        status,body=self.request(max_tokens=11)
        self.assertEqual(status,200,body);self.assertEqual(harness.JOBS[-1]['max_new_tokens'],11)

    def test_explicit_controls_and_seed_reach_job(self):
        status, body = self.request(temperature=.8, top_p=.9, top_k=12, min_p=.02,
                                    presence_penalty=.2, frequency_penalty=.3,
                                    repetition_penalty=1.05, seed=42)
        self.assertEqual(status, 200, body)
        job = harness.JOBS[-1]
        self.assertEqual(job['seed'], 42)
        self.assertEqual(job['sampler'], dict(temperature=.8, top_p=.9, top_k=12,
            min_p=.02, presence_penalty=.2, frequency_penalty=.3, repetition_penalty=1.05))

    def test_invalid_controls_are_rejected_before_job(self):
        for key,value in [('top_k',-1),('top_k',1.5),('min_p',2),('seed',True),
                          ('enable_thinking','false'),('frequency_penalty',3)]:
            with self.subTest(key=key,value=value):
                before = len(harness.JOBS)
                status,body = self.request(**{key:value})
                self.assertEqual(status,400,body)
                self.assertEqual(len(harness.JOBS),before)

    def test_native_prefill_and_decode_clocks_are_kept_separate(self):
        events=harness.events(['READY'])
        events[-1].update(time_prefill=.25,time_generate=.5,cached_tokens=1,
                          new_tokens=1,accepted_draft_tokens=2,rejected_draft_tokens=1)
        self.engine.gen=harness.FakeGen(events)
        status,body=self.request()
        self.assertEqual(status,200,body)
        timings=json.loads(body)['timings']
        self.assertEqual(timings['prompt_cached_n'],1)
        self.assertEqual(timings['prompt_ms'],250)
        self.assertEqual(timings['predicted_ms'],500)
        self.assertEqual(timings['prompt_per_second'],4)
        self.assertEqual(timings['predicted_per_second'],2)
        self.assertEqual(timings['accepted_draft_tokens'],2)

    def reasoning_turn(self, text, streaming=False, eos_reason='stop_token', tools=None):
        class ThinkingTok(harness.FakeTok):
            def hf_render_chat_template(self, messages, **kwargs):
                return 'fixture <think>\n'
        self.engine.tok = ThinkingTok()
        self.engine.gen = harness.FakeGen(harness.events([text], eos_reason=eos_reason))
        status, body = self.request(stream=streaming, tools=tools)
        self.assertEqual(status, 200, body)
        if not streaming:
            return json.loads(body)['choices'][0]
        chunks = harness.sse_chunks(body)
        message = {'content': '', 'reasoning_content': ''}
        for chunk in chunks:
            delta = chunk['choices'][0]['delta']
            for key in message:
                message[key] += delta.get(key) or ''
        return {'message': message, 'finish_reason': chunks[-1]['choices'][0]['finish_reason']}

    def test_budget_exhausted_reasoning_stays_separate_from_answer(self):
        thought = 'I am considering the geometry and bicycle motion before writing the page.'
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                choice = self.reasoning_turn(thought, streaming, eos_reason='max_new_tokens')
                self.assertEqual(choice['message']['content'], '')
                self.assertEqual(choice['message']['reasoning_content'], thought)
                self.assertEqual(choice['finish_reason'], 'length')

    def test_complete_reasoning_and_answer_match_in_both_response_modes(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                choice = self.reasoning_turn('Consider the request.</think>READY', streaming)
                self.assertEqual(choice['message']['content'], 'READY')
                self.assertEqual(choice['message']['reasoning_content'], 'Consider the request.')
                self.assertEqual(choice['finish_reason'], 'stop')

    def test_implicit_tool_transition_keeps_preceding_thought_out_of_content(self):
        tools = [{'type':'function','function':{'name':'write','parameters':{'type':'object',
            'properties':{'content':{'type':'string'}}}}}]
        thought = 'The file is ready. I will write it now.'
        markup = '<tool_call>\n<function=write>\n<parameter=content>\n  READY\n\n</parameter>\n</function>\n</tool_call>'
        choice = self.reasoning_turn(thought + markup, tools=tools)
        self.assertIsNone(choice['message']['content'])
        self.assertEqual(choice['message']['reasoning_content'], thought)
        self.assertEqual(json.loads(choice['message']['tool_calls'][0]['function']['arguments']),
                         {'content': '  READY\n'})

if __name__ == '__main__':
    unittest.main()
