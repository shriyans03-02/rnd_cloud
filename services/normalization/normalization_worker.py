import threading
import time
import traceback

class NormalizationWorker:
    def __init__(self, service, poll_interval: float = 5.0):
        self.service = service
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        print("------------------- Normalization Worker ------------------------------------")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self):
        print("[+++++++++++++++NORMALIZER] loop started")
        while not self._stop.is_set():
            print("[+++++++++++++++NORMALIZER] tick")
            try:
                table = self.service.pick_next_table()
                print("[+++++++++++++++NORMALIZER] picked:", table)
                if table:
                    self.service.normalize_table(table)
            except Exception as e:
                print("[+++++++++++++++NORMALIZER] error:", e)
                print(traceback.format_exc())

            time.sleep(self.poll_interval)
