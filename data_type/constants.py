# POS_TAG_RELATED
USE_BROWN_CORPUS = True
USE_TREEBANK_CORPUS = False
USE_CONLL_CORPUS = False


class Constants:
    UNROLL_RNN = False
    MODE = 2

    @staticmethod
    def enableUnrollMode():
        Constants.UNROLL_RNN = True

    @staticmethod
    def disableUnrollMode():
        Constants.UNROLL_RNN = False

    @staticmethod
    def fineGrainedMode():
        Constants.MODE = 1

    @staticmethod
    def wholeFeatureMode():
        Constants.MODE = 2
