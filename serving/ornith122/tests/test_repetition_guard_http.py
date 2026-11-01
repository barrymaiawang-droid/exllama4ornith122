#!/usr/bin/env python3
"""CPU test for the v3 serving bundle's opt-in `repetition_guard` request field.

Loads the patched ornith_serve.py straight out of the staged bundle and drives the real
ThreadingHTTPServer with fakes - no torch, no GPU, no model - so the thing under test is the
HTTP contract itself: validation at the edge, the Job kwargs that actually reach the engine,
and the guard summary on both response shapes.

Run: python3 tests/test_repetition_guard_http.py
"""
import importlib.util
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1] / 'bundle'
SERIAL = 7
V2_JOB_KWARGS = {'input_ids', 'max_new_tokens', 'min_new_tokens', 'stop_conditions',
                 'sampler', 'decode_special_tokens'}

JOBS = []
CANCELLED = []


class FakeTok:
    """Just enough tokenizer for the engine: render, encode, stop ids."""

    def single_id(self, name):
        return None                      # no stop ids -> Job.stop_conditions is None

    def encode(self, text, encode_special_tokens=False):
        return T(max(1, len(text.split())))

    def hf_render_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        # These guard fixtures replay answer text, so render the native
        # generation prefix with thinking already closed. Reasoning fixtures
        # provide an explicit opening prefix in test_generation_contract.py.
        return " | ".join((m.get("content") or "") for m in messages) + "</think>"


class T:
    """Token-count stand-in: the server only reads shape[-1]."""

    def __init__(self, n):
        self.shape = (1, n)

    def __len__(self):
        return self.shape[-1]


class FakeJob:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeBackend:
    @staticmethod
    def Job(**kwargs):                   # noqa: N802 - mirrors the pinned Job factory
        JOBS.append(kwargs)
        return FakeJob(**kwargs)

    @staticmethod
    def ArgmaxSampler():
        return object()

    @staticmethod
    def TopPSampler(**kwargs):
        return object()

    @staticmethod
    def runtime_info():
        return {}


class FakeGen:
    """Replays a scripted turn; records what was enqueued and cancelled."""

    def __init__(self, events):
        self.events = list(events)

    def enqueue(self, job):
        return SERIAL

    def iterate(self):
        return iter(self.events)

    def cancel(self, job):
        CANCELLED.append(job)


def load_module():
    spec = importlib.util.spec_from_file_location('ornith_serve_v3', BUNDLE / 'ornith_serve.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def events(texts, guard=None, eos_reason='stop_token'):
    out = [{'serial': SERIAL, 'stage': 'streaming', 'text': t} for t in texts]
    final = {'serial': SERIAL, 'eos': True, 'eos_reason': eos_reason}
    if guard is not None:
        final['repetition_guard'] = guard
    out.append(final)
    return out


GUARD_SUMMARY = {'mode': 'active', 'detections': 1, 'interventions': [{'tier': 3, 'position': 40}],
                 'tier': 3, 'rollback_tokens': 12, 'injected_tokens': 0, 'visible_new_tokens': 42}


def start_server(mod):
    engine = mod.Engine(FakeTok(), FakeGen([]), FakeBackend(), ctx=4096, ndt=2,
                        served_name='guard-test')
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), mod.make_handler(engine))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return engine, httpd, httpd.server_address[1]


def post(port, body):
    req = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % port,
                                 json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def sse_chunks(text):
    out = []
    for line in text.splitlines():
        if line.startswith('data: '):
            payload = line[6:]
            if payload != '[DONE]':
                out.append(json.loads(payload))
    return out


FAILS = []


def check(name, cond, detail=''):
    print(('  ok   ' if cond else '  FAIL ') + name + ('' if cond else '   ' + str(detail)[:160]))
    if not cond:
        FAILS.append(name)


def main():
    mod = load_module()
    engine, httpd, port = start_server(mod)
    base = {'model': 'guard-test', 'messages': [{'role': 'user', 'content': 'hi'}],
            'max_tokens': 16}

    print('-- unopted requests must be the v2 request')
    for label, extra in (('absent', {}), ('false', {'repetition_guard': False}),
                         ('null', {'repetition_guard': None})):
        JOBS.clear()
        engine.gen = FakeGen(events(['Alpha ', 'beta']))
        st, body = post(port, dict(base, **extra))
        j = json.loads(body)
        check('%s: 200' % label, st == 200, (st, body[:120]))
        check('%s: Job kwargs are exactly the v2 set' % label,
              set(JOBS[-1]) == V2_JOB_KWARGS, sorted(JOBS[-1]))
        check('%s: no repetition_guard key in the response' % label,
              'repetition_guard' not in j, sorted(j))
        check('%s: content assembled' % label,
              j['choices'][0]['message']['content'] == 'Alpha beta', j['choices'][0]['message'])
        check('%s: completion_tokens is the visible count' % label,
              j['usage']['completion_tokens'] == 2, j['usage'])

    print('-- opted in: forwarded, and the summary comes back')
    JOBS.clear()
    engine.gen = FakeGen(events(['x'], guard=GUARD_SUMMARY, eos_reason='repetition'))
    st, body = post(port, dict(base, repetition_guard=True))
    j = json.loads(body)
    check('true: Job got repetition_guard=True', JOBS[-1].get('repetition_guard') is True, JOBS[-1])
    check('true: only that kwarg was added',
          set(JOBS[-1]) == V2_JOB_KWARGS | {'repetition_guard'}, sorted(JOBS[-1]))
    check('true: summary surfaced on the body', j.get('repetition_guard') == GUARD_SUMMARY,
          j.get('repetition_guard'))

    opts = {'mode': 'shadow', 'min_match': 8, 'confirm': 32, 'max_interventions': 2,
            'escalate_within': 64, 'nudge': None}
    JOBS.clear()
    engine.gen = FakeGen(events(['x']))
    st, body = post(port, dict(base, repetition_guard=opts))
    check('object: forwarded as given', JOBS[-1].get('repetition_guard') == opts,
          JOBS[-1].get('repetition_guard'))
    JOBS.clear()
    engine.gen = FakeGen(events(['x']))
    st, body = post(port, dict(base, repetition_guard={'mode': 'shadow'}))
    check('object partial: only the given key is forwarded',
          JOBS[-1].get('repetition_guard') == {'mode': 'shadow'}, JOBS[-1].get('repetition_guard'))
    JOBS.clear()
    engine.gen = FakeGen(events(['x']))
    st, body = post(port, dict(base, repetition_guard={}))
    check('object empty: forwarded as {}', JOBS[-1].get('repetition_guard') == {},
          JOBS[-1].get('repetition_guard'))


    print('-- malformed options are refused at the edge, before any Job exists')
    bad = [('string', 'yes'), ('number', 3), ('list', [1]),
           ('unknown key', {'min_match': 8, 'minmatch': 8}), ('bad mode', {'mode': 'aggressive'}),
           ('min_match 1', {'min_match': 1}), ('min_match bool', {'min_match': True}),
           ('confirm 0', {'confirm': 0}), ('max_interventions -1', {'max_interventions': -1}),
           ('escalate_within -1', {'escalate_within': -1}), ('nudge int', {'nudge': 5}),
           ('min_match float', {'min_match': 2.5})]
    for label, value in bad:
        JOBS.clear()
        engine.gen = FakeGen(events(['x']))
        st, body = post(port, dict(base, repetition_guard=value))
        j = json.loads(body)
        check('invalid (%s): 400 invalid_request_error' % label,
              st == 400 and j.get('error', {}).get('type') == 'invalid_request_error',
              (st, body[:120]))
        check('invalid (%s): no Job constructed' % label, not JOBS, JOBS)

    print('-- streaming')
    JOBS.clear()
    engine.gen = FakeGen(events(['Alpha ', 'beta'], guard=GUARD_SUMMARY, eos_reason='repetition'))
    st, body = post(port, dict(base, stream=True, repetition_guard=True))
    chunks = sse_chunks(body)
    last = chunks[-1] if chunks else {}
    check('stream: 200 with chunks', st == 200 and bool(chunks), (st, body[:120]))
    check('stream: summary on the final chunk', last.get('repetition_guard') == GUARD_SUMMARY, last)
    check('stream: usage is the visible count',
          last.get('usage', {}).get('completion_tokens') == 2, last.get('usage'))
    check('stream: finish_reason is stop', last.get('choices', [{}])[0].get('finish_reason') == 'stop',
          last.get('choices'))
    check('stream: visible text arrived',
          ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks) == 'Alpha beta',
          body[:200])

    JOBS.clear()
    engine.gen = FakeGen(events(['Alpha']))
    st, body = post(port, dict(base, stream=True))
    chunks = sse_chunks(body)
    check('stream unopted: no summary on any chunk',
          all('repetition_guard' not in c for c in chunks), chunks[-1] if chunks else body[:120])
    check('stream unopted: Job kwargs are the v2 set', set(JOBS[-1]) == V2_JOB_KWARGS,
          sorted(JOBS[-1]))

    print('-- /v1/completions takes the same field')
    JOBS.clear()
    engine.gen = FakeGen(events(['x']))
    req = urllib.request.Request('http://127.0.0.1:%d/v1/completions' % port,
                                 json.dumps({'model': 'guard-test', 'prompt': 'hi',
                                             'max_tokens': 16,
                                             'repetition_guard': {'mode': 'shadow'}}).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        st = r.status
    check('/v1/completions: 200', st == 200, st)
    check('/v1/completions: forwarded', JOBS[-1].get('repetition_guard') == {'mode': 'shadow'},
          JOBS[-1].get('repetition_guard'))

    print('-- fleet default (REPETITION_GUARD): what an unmentioned field means')
    engine.guard_default = True
    JOBS.clear()
    engine.gen = FakeGen(events(['x'], guard=GUARD_SUMMARY, eos_reason='repetition'))
    st, body = post(port, dict(base))
    j = json.loads(body)
    check('default on, no field: Job got repetition_guard=True',
          JOBS[-1].get('repetition_guard') is True, JOBS[-1])
    check('default on, no field: summary surfaced', j.get('repetition_guard') == GUARD_SUMMARY,
          j.get('repetition_guard'))

    ZERO = {'mode': 'active', 'detections': 0, 'interventions': [], 'tier': 0,
            'rollback_tokens': 0, 'injected_tokens': 0, 'visible_new_tokens': 1}
    JOBS.clear()
    engine.gen = FakeGen(events(['x'], guard=ZERO))
    st, body = post(port, dict(base))
    j = json.loads(body)
    check('default on, no field, nothing detected: no summary key (the v2 body shape)',
          'repetition_guard' not in j, sorted(j))

    JOBS.clear()
    engine.gen = FakeGen(events(['x'], guard=ZERO))
    st, body = post(port, dict(base, repetition_guard=True))
    j = json.loads(body)
    check('explicit true, nothing detected: the all-zero summary is still returned',
          j.get('repetition_guard') == ZERO, j.get('repetition_guard'))

    JOBS.clear()
    engine.gen = FakeGen(events(['x'], guard=ZERO))
    st, body = post(port, dict(base, stream=True))
    chunks = sse_chunks(body)
    check('default on, streaming, nothing detected: no summary on the final chunk',
          all('repetition_guard' not in c for c in chunks), chunks[-1] if chunks else body[:120])

    JOBS.clear()
    engine.gen = FakeGen(events(['x']))
    st, body = post(port, dict(base, repetition_guard=False))
    check('default on, explicit false: guard off for that one request',
          set(JOBS[-1]) == V2_JOB_KWARGS, sorted(JOBS[-1]))

    engine.guard_default = {'mode': 'shadow'}
    JOBS.clear()
    engine.gen = FakeGen(events(['x']))
    st, body = post(port, dict(base))
    check('default shadow, no field: forwarded as shadow',
          JOBS[-1].get('repetition_guard') == {'mode': 'shadow'}, JOBS[-1].get('repetition_guard'))

    engine.guard_default = True
    JOBS.clear()
    engine.gen = FakeGen(events(['x']))
    st, body = post(port, dict(base, repetition_guard={'mode': 'shadow'}))
    check('default on, request asks shadow: the request wins',
          JOBS[-1].get('repetition_guard') == {'mode': 'shadow'}, JOBS[-1].get('repetition_guard'))

    with urllib.request.urlopen('http://127.0.0.1:%d/props' % port, timeout=30) as r:
        props = json.loads(r.read().decode())
    check('/props reports the default', props.get('repetition_guard_default') is True,
          props.get('repetition_guard_default'))
    engine.guard_default = None
    with urllib.request.urlopen('http://127.0.0.1:%d/props' % port, timeout=30) as r:
        props = json.loads(r.read().decode())
    check('/props omits the default when unset', 'repetition_guard_default' not in props,
          sorted(props))

    print('-- REPETITION_GUARD parsing')
    saved = os.environ.get('REPETITION_GUARD')
    for raw, want in ((None, None), ('', None), ('0', None), ('false', None), ('off', None),
                      ('1', True), ('true', True), ('on', True), ('shadow', {'mode': 'shadow'}),
                      ('{"mode":"shadow","min_match":4}', {'mode': 'shadow', 'min_match': 4})):
        if raw is None:
            os.environ.pop('REPETITION_GUARD', None)
        else:
            os.environ['REPETITION_GUARD'] = raw
        got = mod._guard_default()
        check('REPETITION_GUARD=%r -> %r' % (raw, want), got == want, got)
    for bad in ('nonsense', '{"min_match":1}', '{"unknown":1}'):
        os.environ['REPETITION_GUARD'] = bad
        try:
            mod._guard_default()
            check('REPETITION_GUARD=%r rejected at startup' % bad, False, 'no SystemExit')
        except SystemExit:
            check('REPETITION_GUARD=%r rejected at startup' % bad, True)
    if saved is None:
        os.environ.pop('REPETITION_GUARD', None)
    else:
        os.environ['REPETITION_GUARD'] = saved

    httpd.shutdown()
    print()
    if FAILS:
        print('FAILED %d check(s): %s' % (len(FAILS), ', '.join(FAILS)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
