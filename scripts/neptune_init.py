import neptune

# Inizializza Neptune
def init_neptune():
# Inizializza Neptune
    run = neptune.init_run(
        project="famato/tesi",  # Sostituisci con il tuo project name
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjcxZDE1OC01YmVjLTRiZGUtYTZhYi01YzdiM2MxNTcxMjIifQ=="            # Sostituisci con il tuo API token
    )
    return run
