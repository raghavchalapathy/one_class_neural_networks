from sklearn.preprocessing import StandardScaler
import numpy as np

class CreateDataSet:
    
    dataPath = '../data/raw/'
    scaler = StandardScaler()


    def get_TestingData(self):

        
        import tempfile
        import pickle
       
        with open(self.dataPath+'usps_data.pkl','rb') as fp:
              loaded_data1 = pickle.load(fp, encoding='latin1')

        labels = loaded_data1['target']
        data = loaded_data1['data']
        
        ## Scale the data 
        print(self.scaler.fit(data))
        StandardScaler(copy=True, with_mean=True, with_std=True)
        data = self.scaler.transform(data)
   
        ## Select Ones
        k_ones = np.where(labels == 2)
        label_ones = labels[k_ones]
        data_ones = data[k_ones]

        k_sevens = np.where(labels == 8)
        label_sevens = labels[k_sevens]
        data_sevens = data[k_sevens]


        data_ones = data_ones[220:440] 
        data_sevens = data_sevens[0:11] 

        
        label_ones      =  1*np.ones(len(data_ones))
        label_sevens    =  np.zeros(len(data_sevens))
        


        return [data_ones,label_ones,data_sevens,label_sevens]

    def get_TestingData_With_Random_UniformNoise(self):

        
        import tempfile
        import pickle
       
        with open(self.dataPath+'usps_data.pkl','rb') as fp:
              loaded_data1 = pickle.load(fp, encoding='latin1')

        labels = loaded_data1['target']
        data = loaded_data1['data']
        
        ## Scale the data 
        print(self.scaler.fit(data))
        StandardScaler(copy=True, with_mean=True, with_std=True)
        data = self.scaler.transform(data)
   
        ## Select Ones
        k_ones = np.where(labels == 2)
        label_ones = labels[k_ones]
        data_ones = data[k_ones]

        k_sevens = np.where(labels == 8)
        label_sevens = labels[k_sevens]
        data_sevens = data[k_sevens]


        data_ones = data_ones[220:440] 
        data_sevens =  np.random.uniform(0,1,(len(data_ones),256))

        
        label_ones      =  1*np.ones(len(data_ones))
        label_sevens    =  np.zeros(len(data_sevens))
        


        return [data_ones,label_ones,data_sevens,label_sevens]

    def get_TrainingData(self):
        
        
        import tempfile
        import pickle
       

        with open(self.dataPath+'usps_data.pkl','rb') as fp:
              loaded_data1 = pickle.load(fp, encoding='latin1')

        labels = loaded_data1['target']
        data = loaded_data1['data']
        
        ## Scale the data 
       
        print(self.scaler.fit(data))
        StandardScaler(copy=True, with_mean=True, with_std=True)
        data = self.scaler.transform(data)
   
        ## Select Ones
        k_ones = np.where(labels == 2)
        label_ones = labels[k_ones]
        data_ones = data[k_ones]

        k_sevens = np.where(labels == 8)
        label_sevens = labels[k_sevens]
        data_sevens = data[k_sevens]


        data_ones = data_ones[:220] 
        label_ones      =  1*np.ones(len(data_ones))
       
        return [data_ones,label_ones] 
    
    def get_FAKE_Noise_TrainingData(self,X):
        
        data_noise =  np.random.uniform(0,1,(len(X),256))
        label_noise    =  np.zeros(len(data_noise))
        return [data_noise,label_noise]
