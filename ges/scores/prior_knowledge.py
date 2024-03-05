'''
LUCAS = {
    0: "Smoking",
    1: "Yellow_Fingers",
    2: "Anxiety",
    3: "Peer_Pressure",
    4: "Genetics",
    5: "Attention_Disorder",
    6: "Born_an_Even_Day",
    7: "Car_Accident",
    8: "Fatigue",
    9: "Allergy",
    10: "Coughing",
    11: "Lung_Cancer"
}

Asia = {
    0: 'VisitAsia', 
    1: 'TB', 
    2: 'Smoking', 
    3: 'Cancer', 
    4: 'TBorCancer', 
    5: 'XRay', 
    6: 'Bronchitis', 
    7: 'Dysphenia'
}

SACHS = {
    0: 'praf',
    1: 'pmek',
    2: 'plcg',
    3: 'PIP2',
    4: 'PIP3',
    5: 'p44/42',
    6: 'pakts473',
    7: 'PKA',
    8: 'PKC',
    9: 'P38',
    10: 'pjnk'
}

'''

import numpy as np


class PriorKnowledge:
    def __init__(self):
        self.prior_knowledge = {}

    def add_LUCAS_knowledge(self):
        self.prior_knowledge['LUCAS'] = {}

        # GPT3 knowledge
        self.prior_knowledge['LUCAS']['GPT3'] = np.ones((12, 12)) * 2
        self.prior_knowledge['LUCAS']['GPT3'][2][0] = 1
        self.prior_knowledge['LUCAS']['GPT3'][3][0] = 1
        self.prior_knowledge['LUCAS']['GPT3'][0][1] = 1
        self.prior_knowledge['LUCAS']['GPT3'][0][11] = 1
        self.prior_knowledge['LUCAS']['GPT3'][4][11] = 1
        self.prior_knowledge['LUCAS']['GPT3'][9][0] = 1
        self.prior_knowledge['LUCAS']['GPT3'][10][11] = 1
        self.prior_knowledge['LUCAS']['GPT3'][8][0] = 1
        self.prior_knowledge['LUCAS']['GPT3'][5][0] = 1


        # GPT4 knowledge
        self.prior_knowledge['LUCAS']['GPT4'] = np.ones((12, 12)) * 2
        self.prior_knowledge['LUCAS']['GPT4'][3][0] = 1
        self.prior_knowledge['LUCAS']['GPT4'][0][1] = 1
        self.prior_knowledge['LUCAS']['GPT4'][0][11] = 1
        self.prior_knowledge['LUCAS']['GPT4'][4][11] = 1
        self.prior_knowledge['LUCAS']['GPT4'][0][10] = 1
        self.prior_knowledge['LUCAS']['GPT4'][5][0] = 1

        # Gemini knowledge
        self.prior_knowledge['LUCAS']['Gemini'] = np.ones((12, 12)) * 2
        self.prior_knowledge['LUCAS']['Gemini'][3][0] = 1
        self.prior_knowledge['LUCAS']['Gemini'][0][1] = 1
        self.prior_knowledge['LUCAS']['Gemini'][4][11] = 1
        self.prior_knowledge['LUCAS']['Gemini'][11][10] = 1
        self.prior_knowledge['LUCAS']['Gemini'][11][8] = 1

    def add_Asia_knowledge(self):
        self.prior_knowledge['Asia'] = {}

        # GPT3 knowledge
        self.prior_knowledge['Asia']['GPT3'] = np.ones((8, 8)) * 2
        self.prior_knowledge['Asia']['GPT3'][2][3] = 1
        self.prior_knowledge['Asia']['GPT3'][3][4] = 1
        self.prior_knowledge['Asia']['GPT3'][4][5] = 1
        self.prior_knowledge['Asia']['GPT3'][4][6] = 1
        self.prior_knowledge['Asia']['GPT3'][4][7] = 1
        self.prior_knowledge['Asia']['GPT3'][1][4] = 1
        self.prior_knowledge['Asia']['GPT3'][6][7] = 1
        self.prior_knowledge['Asia']['GPT3'][7][5] = 1

        # GPT4 knowledge
        self.prior_knowledge['Asia']['GPT4'] = np.ones((8, 8)) * 2
        self.prior_knowledge['Asia']['GPT4'][0][1] = 1
        self.prior_knowledge['Asia']['GPT4'][1][4] = 1
        self.prior_knowledge['Asia']['GPT4'][2][3] = 1
        self.prior_knowledge['Asia']['GPT4'][2][6] = 1
        self.prior_knowledge['Asia']['GPT4'][3][4] = 1
        self.prior_knowledge['Asia']['GPT4'][4][5] = 1
        self.prior_knowledge['Asia']['GPT4'][4][7] = 1
        self.prior_knowledge['Asia']['GPT4'][6][7] = 1

        # Gemini knowledge
        self.prior_knowledge['Asia']['Gemini'] = np.ones((8, 8)) * 2
        self.prior_knowledge['Asia']['Gemini'][1][4] = 1
        self.prior_knowledge['Asia']['Gemini'][2][3] = 1
        self.prior_knowledge['Asia']['Gemini'][3][4] = 1
        self.prior_knowledge['Asia']['Gemini'][3][7] = 1
        self.prior_knowledge['Asia']['Gemini'][6][7] = 1


        pass

    def add_SACHS_knowledge(self):
        self.prior_knowledge['SACHS'] = {}

        # GPT3 knowledge
        self.prior_knowledge['SACHS']['GPT3'] = np.ones((11, 11)) * 2
        pass