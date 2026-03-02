try:
    import neptune
except Exception:
    neptune = None


class _NoOpChannel:
    def log(self, *args, **kwargs):
        return None

    def append(self, *args, **kwargs):
        return None

    def upload(self, *args, **kwargs):
        return None


class _NoOpRun:
    def __getitem__(self, key):
        return _NoOpChannel()

    def __setitem__(self, key, value):
        return None

    def stop(self):
        return None


def init_neptune():
    if neptune is None:
        print("[WARN] Neptune non disponibile: logging disabilitato (no-op).")
        return _NoOpRun()

    run = neptune.init_run(
        project="",  # Sostituisci con il tuo project name
        api_token=""  # Sostituisci con il tuo API token
    )
    return run
