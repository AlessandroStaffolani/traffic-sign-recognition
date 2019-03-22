import numpy as np

from src.init import init_directories
from src.utility.system_utility import get_arguments, welcome_msg


def main(args):
    print(args)
    try:
        from src.controllers.MenuController import MenuController
        from src.controllers.MultipleRunController import MultipleRunController
    except FileNotFoundError:
        init_directories()
        from src.controllers.MenuController import MenuController
        from src.controllers.MultipleRunController import MultipleRunController

    np.random.seed(args.random_seed)

    welcome_msg(args.random_seed)

    if args.run_file is None:
        # single execution in interactive or script mode
        menu = MenuController(mode=args.mode, actions=args.actions, model=args.model_code, batch_size=args.batch_size,
                              epochs=args.epochs,
                              image_shape=args.image_shape, num_workers=args.n_workers, model_path=args.model_file,
                              weights_path=args.weights_file, color_mode=args.color_mode, split_factor=args.split_factor,
                              n_train_samples=args.n_train_samples, n_validation_samples=args.n_validation_samples,
                              train_dir=args.train_dir, validation_dir=args.validation_dir, test_dir=args.test_dir,
                              use_augmentation=args.use_augmentation)
    else:
        # multiple execution in script mode
        multiple = MultipleRunController(args.run_file)
        multiple.execute()

    # df = pd.read_csv('data/training/training_table.csv')
    # print(df.info())


if __name__ == '__main__':
    args = get_arguments()
    main(args)
