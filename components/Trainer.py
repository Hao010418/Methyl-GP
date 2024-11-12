from components import DataManager, ModelManager, IOManager


class Trainer():
    def __init__(self, config):
        self.config = config
        self.IOM = IOManager.IOManager(self)
        self.DM = DataManager.DataManager(self)
        self.MM = ModelManager.ModelManager(self)

    def setIO(self):
        self.IOM.initialize()
        self.IOM.log.Info('Set IO Over.')

    def setVisualization(self):
        pass

    def load_data(self):
        self.DM.load_data()
        self.IOM.log.Info('Load Data Over')

    def init_model(self):
        self.MM.init_model()
        self.IOM.log.Info('Init Model Over.')

    def check_model(self):
        self.MM.check_model()
        self.IOM.log.Info('Check Model Over.')

    # def choose_optimizer(self):
    #     self.MM.choose_optimizer()
    #     self.IOM.log.Info('Choose Optimizer Over.')

    def choose_loss_function(self):
        self.MM.choose_loss_function()
        self.IOM.log.Info('Choose Loss Function Over.')

    def train_model(self):
        self.IOM.log.Info('Train Model Start.')
        self.IOM.log.Info(f'Config: {self.config}')
        self.MM.train_part()
        # draw experiments pictures

        self.IOM.log.Info('Train Model Over.')
