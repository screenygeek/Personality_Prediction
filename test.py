from utils import *

class test:
    def __init__(self,df,url):
        self.model_names = glob.glob(url + "*.sav")
        self.all_models = dict()
        for i in self.model_names:
            model_name = i.split('\\')[-1]
            model_name = model_name.split('.')[0]
            self.all_models[model_name] = joblib.load(i)
        self.df = df
        self._preprocessing()
    def _preprocessing(self):
        preprocessing_obj = preprocessing(self.df)
        self.df = preprocessing_obj.fit_transform()
        print(self.df)
    def get_predictions(self,vectorizer):
        predictions = dict()
        if len(self.df) > 0:
            names = {'ie':{0:'Introvert', 1: 'Extrovert'},'ft':{0: 'Thinker',1:'Feeler'},'ns':{0:'Sensor',1:'Intuitive'},'pj':{0:'Judger',1:'Perceiver'}}
            tfs = vectorizer.transform(self.df['preprocessed_text'])
            for i in self.all_models.keys():
                y_pred = self.all_models[i].predict(tfs)
                print(y_pred)
                predictions[i] = names[i][y_pred[0]]
                #print(predictions[i])
            data = dict()
            for i in self.all_models.keys():
                data[i] = predictions[i]
            return data,200
        else:
            return data,400
        
        