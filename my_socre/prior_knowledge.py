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
    def __init__(self, dataset, LLMs = ['GPT3', 'GPT4', 'Gemini']):
        self.prior_knowledge = {}
        if dataset not in ['LUCAS', 'Asia', 'SACHS', 'Survey']:
            raise ValueError("The dataset should be one of the following: ['LUCAS', 'Asia', 'SACHS']")
        

        self.LLMs = LLMs
        if dataset == 'LUCAS':
            self.add_LUCAS_knowledge()
        if dataset == 'Asia':
            self.add_Asia_knowledge()
        if dataset == 'SACHS':
            self.add_SACHS_knowledge()
        if dataset == 'Survey':
            self.add_Survey_knowledge()
            self.add_Survey_true_knowledge()

        self.intersection_result = {}
        self.add_intersection_result()

    def add_intersection_result(self):
        for dataset in self.prior_knowledge:
            p = 0
            edge_sets = []
            for method in self.prior_knowledge[dataset]:
                edge_sets.append(set(zip(*np.where(self.prior_knowledge[dataset][method] == 1))))
                p = self.prior_knowledge[dataset][method].shape[0]

            re = edge_sets[0].intersection(*edge_sets[1:])
            result_matrix = np.zeros((p, p))
            for edge in re:
                result_matrix[edge] = 1
            self.intersection_result[dataset] = result_matrix

    def add_Earthquake_knowledge(self):
        self.prior_knowledge['Earthquake'] = {}
        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['Earthquake']['GPT3'] = np.zeros((5, 5))
            # Burglary → Alarm
            self.prior_knowledge['Earthquake']['GPT3'][0][1] = 1
            # Earthquake → Alarm
            self.prior_knowledge['Earthquake']['GPT3'][2][1] = 1
            # Alarm → JohnCalls
            self.prior_knowledge['Earthquake']['GPT3'][1][3] = 1
            # Alarm → MaryCalls
            self.prior_knowledge['Earthquake']['GPT3'][1][4] = 1

        if 'GPT4' in self.LLMs:
            # Initialize the matrix for 'EarthquakeModel'
            self.prior_knowledge['Earthquake']['GPT4'] = np.zeros((5, 5))
            # Burglary → Alarm
            self.prior_knowledge['Earthquake']['GPT4'][0][1] = 1
            # Earthquake → Alarm
            self.prior_knowledge['Earthquake']['GPT4'][2][1] = 1
            # Alarm → JohnCalls
            self.prior_knowledge['Earthquake']['GPT4'][1][3] = 1
            # Alarm → MaryCalls
            self.prior_knowledge['Earthquake']['GPT4'][1][4] = 1

        if 'Gemini' in self.LLMs:
            self.prior_knowledge['Earthquake'] ['Gemini'] = np.zeros((5,5))
            # Earthquake → Alarm (if earthquake alarm system exists)
            self.prior_knowledge['Earthquake']['Gemini'][2][1] = 1  # Earthquake -> Alarm
        

    def add_Survey_true_knowledge(self):
        self.prior_knowledge['Survey']['True'] = np.zeros((6, 6))
        self.prior_knowledge['Survey']['True'][0][1] = 1
        self.prior_knowledge['Survey']['True'][2][1] = 1
        self.prior_knowledge['Survey']['True'][1][3] = 1
        self.prior_knowledge['Survey']['True'][1][4] = 1
        self.prior_knowledge['Survey']['True'][3][5] = 1
        self.prior_knowledge['Survey']['True'][4][5] = 1

    # A E S O R T
    def add_Survey_knowledge(self):
        self.prior_knowledge['Survey'] = {}
        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['Survey']['GPT3'] = np.ones((6, 6)) * 2
            self.prior_knowledge['Survey']['GPT3'][0][3] = 1
            self.prior_knowledge['Survey']['GPT3'][1][3] = 1
            self.prior_knowledge['Survey']['GPT3'][2][3] = 1
            self.prior_knowledge['Survey']['GPT3'][4][5] = 1
            self.prior_knowledge['Survey']['GPT3'][0][5] = 1
        
        if 'GPT4' in self.LLMs:
            # GPT4 knowledge
            self.prior_knowledge['Survey']['GPT4'] = np.ones((6, 6)) * 2
            self.prior_knowledge['Survey']['GPT4'][2][3] = 1
            self.prior_knowledge['Survey']['GPT4'][3][4] = 1

        if 'Gemini' in self.LLMs:
            # Gemini knowledge
            self.prior_knowledge['Survey']['Gemini'] = np.ones((6, 6)) * 2
            self.prior_knowledge['Survey']['Gemini'][1][3] = 1
            self.prior_knowledge['Survey']['Gemini'][4][3] = 1


    def add_LUCAS_knowledge(self):
        self.prior_knowledge['LUCAS'] = {}

        if 'GPT3' in self.LLMs:
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

        if 'GPT4' in self.LLMs:
            # GPT4 knowledge
            self.prior_knowledge['LUCAS']['GPT4'] = np.ones((12, 12)) * 2
            self.prior_knowledge['LUCAS']['GPT4'][3][0] = 1
            self.prior_knowledge['LUCAS']['GPT4'][0][1] = 1
            self.prior_knowledge['LUCAS']['GPT4'][0][11] = 1
            self.prior_knowledge['LUCAS']['GPT4'][4][11] = 1
            self.prior_knowledge['LUCAS']['GPT4'][0][10] = 1
            self.prior_knowledge['LUCAS']['GPT4'][5][0] = 1

        if 'Gemini' in self.LLMs:
            # Gemini knowledge
            self.prior_knowledge['LUCAS']['Gemini'] = np.ones((12, 12)) * 2
            self.prior_knowledge['LUCAS']['Gemini'][3][0] = 1
            self.prior_knowledge['LUCAS']['Gemini'][0][1] = 1
            self.prior_knowledge['LUCAS']['Gemini'][4][11] = 1
            self.prior_knowledge['LUCAS']['Gemini'][11][10] = 1
            self.prior_knowledge['LUCAS']['Gemini'][11][8] = 1

    def add_Asia_knowledge(self):
        self.prior_knowledge['Asia'] = {}

        if 'GPT3' in self.LLMs:
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

        if 'GPT4' in self.LLMs:
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


        if 'Gemini' in self.LLMs:
            # Gemini knowledge
            self.prior_knowledge['Asia']['Gemini'] = np.ones((8, 8)) * 2
            self.prior_knowledge['Asia']['Gemini'][1][4] = 1
            self.prior_knowledge['Asia']['Gemini'][2][3] = 1
            self.prior_knowledge['Asia']['Gemini'][3][4] = 1
            self.prior_knowledge['Asia']['Gemini'][3][7] = 1
            self.prior_knowledge['Asia']['Gemini'][6][7] = 1

    def add_SACHS_knowledge(self):
        self.prior_knowledge['SACHS'] = {}

        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['SACHS']['GPT3'] = np.ones((11, 11)) * 2
            self.prior_knowledge['SACHS']['GPT3'][2][8] = 1
            self.prior_knowledge['SACHS']['GPT3'][2][3] = 1
            self.prior_knowledge['SACHS']['GPT3'][3][4] = 1
            self.prior_knowledge['SACHS']['GPT3'][4][6] = 1
            self.prior_knowledge['SACHS']['GPT3'][4][7] = 1
            self.prior_knowledge['SACHS']['GPT3'][5][6] = 1
            self.prior_knowledge['SACHS']['GPT3'][5][10] = 1
            self.prior_knowledge['SACHS']['GPT3'][6][9] = 1
            self.prior_knowledge['SACHS']['GPT3'][7][5] = 1
            self.prior_knowledge['SACHS']['GPT3'][7][6] = 1
            self.prior_knowledge['SACHS']['GPT3'][7][9] = 1
            self.prior_knowledge['SACHS']['GPT3'][8][2] = 1
            self.prior_knowledge['SACHS']['GPT3'][8][5] = 1
            self.prior_knowledge['SACHS']['GPT3'][8][6] = 1
            self.prior_knowledge['SACHS']['GPT3'][8][7] = 1
            self.prior_knowledge['SACHS']['GPT3'][9][10] = 1

        if 'GPT4' in self.LLMs:
            # GPT4 knowledge
            self.prior_knowledge['SACHS']['GPT4'] = np.ones((11, 11)) * 2
            self.prior_knowledge['SACHS']['GPT4'][2][8] = 1
            self.prior_knowledge['SACHS']['GPT4'][2][3] = 1
            self.prior_knowledge['SACHS']['GPT4'][3][4] = 1
            self.prior_knowledge['SACHS']['GPT4'][4][6] = 1
            self.prior_knowledge['SACHS']['GPT4'][0][1] = 1
            self.prior_knowledge['SACHS']['GPT4'][1][5] = 1


        if 'Gemini' in self.LLMs:
            # Gemini knowledge
            self.prior_knowledge['SACHS']['Gemini'] = np.ones((11, 11)) * 2
            self.prior_knowledge['SACHS']['Gemini'][2][3] = 1
            self.prior_knowledge['SACHS']['Gemini'][3][4] = 1
            self.prior_knowledge['SACHS']['Gemini'][4][6] = 1
            self.prior_knowledge['SACHS']['Gemini'][0][1] = 1
            self.prior_knowledge['SACHS']['Gemini'][1][5] = 1
            self.prior_knowledge['SACHS']['Gemini'][8][10] = 1