"""Closing a stream during prefill must cancel the backend at a chunk boundary."""
import json
import socket
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
import test_repetition_guard_http as harness

class SlowPrefill:
    def __init__(self):
        self.cancelled=threading.Event()
        self.finish=threading.Event()
    def enqueue(self,job): return harness.SERIAL
    def iterate(self):
        time.sleep(.05)
        if self.finish.is_set():
            return harness.events(['READY'])
        return [{'serial':harness.SERIAL,'stage':'prefill'}]
    def cancel(self,job): self.cancelled.set();self.finish.set()

class PrefillDisconnect(unittest.TestCase):
    def test_disconnect_releases_engine_before_prefill_finishes(self):
        mod=harness.load_module();mod.KEEPALIVE_S=.05
        gen=SlowPrefill()
        engine=mod.Engine(harness.FakeTok(),gen,harness.FakeBackend(),4096,2,'disconnect')
        server=ThreadingHTTPServer(('127.0.0.1',0),mod.make_handler(engine))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            body=json.dumps({'messages':[{'role':'user','content':'A long prompt'}],'stream':True}).encode()
            client=socket.create_connection(server.server_address)
            client.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
            received=b''
            while b'data: ' not in received:
                received+=client.recv(4096)
            client.shutdown(socket.SHUT_RDWR);client.close()
            self.assertTrue(gen.cancelled.wait(2.5),'Backend kept prefilling after its streaming client disconnected')
        finally:
            gen.finish.set()
            server.shutdown();server.server_close()

if __name__=='__main__': unittest.main()
