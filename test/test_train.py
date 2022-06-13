import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modules.module_for_NER_training import *

if __name__ == '__main__':
    ner_trainer = NerTrainer()
    ner_trainer.run()