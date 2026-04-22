import http.server, os
os.chdir(r'C:\Users\bytevion\Downloads\Byte')

class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a): pass
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

http.server.HTTPServer(('0.0.0.0', 3000), H).serve_forever()
