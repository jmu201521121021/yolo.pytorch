from yolov3.solver.darknet53_solver import TrainDarknet53Solver
from yolov3.configs.default import get_default_config

def setup():
    cfg = get_default_config()
    return  cfg

def trainer():
    cfg = setup()
    trainer = TrainDarknet53Solver(cfg)
    trainer.train()

if __name__ == '__main__':
    trainer()