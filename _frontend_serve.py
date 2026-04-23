import http.server, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a): pass
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

http.server.HTTPServer(('0.0.0.0', 3000), H).serve_forever()
