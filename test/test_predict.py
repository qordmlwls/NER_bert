import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modules.module_for_NER_predict import *


if __name__ == '__main__':
    ner_predicter = NerPrdict()
    out = ner_predicter.run()