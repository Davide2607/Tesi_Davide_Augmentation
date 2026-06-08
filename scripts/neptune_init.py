import os


class _NoOpRun:
    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        return None

    def append(self, value):
        return None

    def log(self, value):
        return None

    def upload(self, value):
        return None

    def stop(self):
        return None


def init_neptune():
    project = os.environ.get("NEPTUNE_PROJECT", "").strip()
    token = os.environ.get("NEPTUNE_API_TOKEN", "").strip()

    if not project or not token:
        print("[NEPTUNE] Neptune non configurato: uso un logger no-op locale.")
        return _NoOpRun()

    try:
        import neptune

        return neptune.init_run(project=project, api_token=token)
    except Exception as e:
        print(f"[NEPTUNE] Impossibile inizializzare Neptune ({e}); uso un logger no-op locale.")
        return _NoOpRun()