import os
import hydra


@hydra.main(config_path="conf", config_name="test_config")
def main(cfg):
    if cfg.Method == "DSB":
        from test_dsb import test
        return test(cfg)
    elif cfg.Method == "DBDSB":
        from test_dbdsb import test
        return test(cfg)
    elif cfg.Method == "RF":
        from test_rf import test
        return test(cfg)
    else: 
        raise NotImplementedError

if __name__ == "__main__":
    main()