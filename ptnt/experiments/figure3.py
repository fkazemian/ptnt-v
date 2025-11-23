from ..io.run import run_from_config, default_config_for_experiment
def main():
    cfg = default_config_for_experiment("figure3")
    run_from_config(cfg)
if __name__ == "__main__":
    main()
