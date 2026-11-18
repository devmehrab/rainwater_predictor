"""
server.py
Lightweight local web server using Python's standard library.
Connects index.html to the numerical methods in simulation.py.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from simulation import (
    run_accumulation_simulation,
    estimate_filling_time,
    estimate_required_intensity,
)

HOST = "127.0.0.1"
PORT = 8000


class PredictorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open("index.html", "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "File not found")

    def do_POST(self):
        if self.path != "/simulate":
            self.send_error(404, "Endpoint not found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        params = json.loads(body.decode("utf-8"))

        # Validate inputs
        area = params.get("area", 0)
        capacity = params.get("capacity", 0)
        v_initial = params.get("v_initial", 0)
        intensity = params.get("intensity", 0)
        eff_percent = params.get("eff_percent", 0)
        duration = params.get("duration", 0)
        step = params.get("step", 0)
        desired_time = params.get("desired_time", 0)

        error_msg = None
        if area <= 0 or capacity <= 0 or duration <= 0 or step <= 0 or desired_time <= 0:
            error_msg = "Area, capacity, duration, step, and desired time must be greater than 0."
        elif v_initial < 0 or v_initial > capacity:
            error_msg = "Initial water volume must be between 0 and tank capacity."
        elif intensity < 0:
            error_msg = "Rainfall intensity cannot be negative."
        elif not (0 <= eff_percent <= 100):
            error_msg = "Collection efficiency must be between 0% and 100%."
        elif step > duration:
            error_msg = "Time step h cannot exceed simulation duration."

        if error_msg:
            self._send_json(400, {"error": error_msg})
            return

        efficiency = eff_percent / 100.0

        # Run your Python numerical methods
        times, v_euler, v_mod = run_accumulation_simulation(
            area, intensity, efficiency, v_initial, duration, step
        )
        fill_time, bisection_hist = estimate_filling_time(
            area, intensity, efficiency, v_initial, capacity
        )
        req_intensity, nr_hist = estimate_required_intensity(
            area, efficiency, v_initial, capacity, desired_time
        )

        final_vol = v_mod[-1]
        collected = final_vol - v_initial
        overflowed = final_vol > capacity
        stored = min(final_vol, capacity)
        excess = max(0.0, final_vol - capacity)
        remaining = max(0.0, capacity - stored)

        response_data = {
            "times": times,
            "v_euler": v_euler,
            "v_mod": v_mod,
            "bisection": {
                "root": fill_time,
                "history": bisection_hist,
                "message": "Tank is already full." if v_initial >= capacity else "Cannot fill within interval."
            },
            "newton": {
                "root": req_intensity,
                "history": nr_hist,
                "message": "Tank is already full." if v_initial >= capacity else "Zero derivative encountered."
            },
            "summary": {
                "capacity": capacity,
                "collected": collected,
                "stored": stored,
                "overflowed": overflowed,
                "excess": excess,
                "remaining": remaining,
                "fill_time": fill_time,
                "req_intensity": req_intensity,
            }
        }
        self._send_json(200, response_data)

    def _send_json(self, status, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), PredictorHandler)
    print(f"Server running! Open your browser at: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()