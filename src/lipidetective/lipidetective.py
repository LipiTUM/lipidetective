import ray
import logging

from src.lipidetective.helpers.utils import parse_config, is_main_process, set_seeds
from src.lipidetective.workflow.trainer import Trainer


def main():
    # 1. PREPARATION
    # Logging configuration
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%d/%m/%Y - %H:%M:%S")

    if is_main_process():
        logging.info('LipiDetective started.')

    # Set seeds for deterministic behavior
    set_seeds()

    # Parse yaml config file
    config, args = parse_config()

    # Set up trainer which will perform the workflow tasks
    trainer = Trainer(config)

    # 2. EXECUTE WORKFLOWS
    if config['model'] == 'random_forest':
        logging.info(f'Random forest run of {config["random_forest"]["type"]} started.')

        trainer.run_random_forest()

    # Hyperparameter Tuning
    elif config['workflow']['tune']:
        logging.info(f"Tuning of {config['model']} network will be performed.")

        if args.head_node_ip is not None:
            logging.info(f'Setting _node_ip_address={args.head_node_ip}')
            ray.init(_node_ip_address=args.head_node_ip)

        trainer.schedule_tuning()

    # Training
    elif config['workflow']['train']:
        if config['workflow']['validate']:
            logging.info(f"Training of {config['model']} network will be performed with validation.")
            trainer.train_with_validation()
        else:
            logging.info(f"Training of {config['model']} network will be performed without validation.")
            trainer.train_without_validation()

    # Testing
    elif config['workflow']['test']:
        logging.info(f"Testing of {config['model']} network will be performed.")
        trainer.test()

    # Predict
    elif config['workflow']['predict']:
        logging.info(f"Prediction using {config['model']} network will be performed.")
        trainer.predict()

    else:
        logging.info('No workflow specified.')

    logging.info('LipiDetective finished.')


if __name__ == "__main__":
    main()
