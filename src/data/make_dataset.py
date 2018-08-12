from sklearn.preprocessing import StandardScaler
import numpy as np
import json

class CreateDataSet:

    dataPath = '../data/raw/'
    scaler = StandardScaler()

    def load_LHCDataset(self,data_dir_path):

        background = np.array([
            json.loads(s)
            for s in open(data_dir_path + '/bgimages_njge3_100k.dat')
        ])
        signal = np.array([
            json.loads(s)
            for s in open(data_dir_path + '/sigimages_njge3_100k.dat')
        ])

        return [background, signal]

    def get_LHC_TrainingData(self, normal, NUM_OF_NORMAL,
                             NUM_OF_ANOMALIES):

        normal = np.reshape(normal,(len(normal),1369))
        

        StandardScaler(copy=True, with_mean=True, with_std=True)
        # self.scaler.fit(normal)
        # normal = self.scaler.transform(normal)
     

        SEED = 42
        NUM_OF_NORMAL_DATA_INSTANCES_TRAIN = NUM_OF_NORMAL
        NUM_OF_ANOMALOUS_DATA_INSTANCES = NUM_OF_ANOMALIES

        np.random.shuffle(normal)
        train_normal = normal[0:NUM_OF_NORMAL_DATA_INSTANCES_TRAIN]
        
        # print("Unique train_normal:train_normal", np.unique(train_normal))
        label_normal = 1 * np.ones(len(train_normal))

        return [train_normal, label_normal]

    def get_LHC_TestingData(self, normal, anomalies, NUM_OF_NORMAL,
                            NUM_OF_ANOMALIES):

        normal = np.reshape(normal,(len(normal),1369))
        anomalies = np.reshape(normal,(len(anomalies),1369))

        StandardScaler(copy=True, with_mean=True, with_std=True)
        # self.scaler.fit(normal)
        # normal = self.scaler.transform(normal)
        # self.scaler.fit(anomalies)
        # anomalies = self.scaler.transform(anomalies)
        
        SEED = 42
        NUM_OF_NORMAL_DATA_INSTANCES_TRAIN = NUM_OF_NORMAL
        NUM_OF_ANOMALOUS_DATA_INSTANCES = NUM_OF_ANOMALIES

        np.random.shuffle(normal)
        np.random.shuffle(anomalies)

        test_normal = normal[NUM_OF_NORMAL:NUM_OF_NORMAL + 5000]
        test_anomalies = anomalies[0:NUM_OF_ANOMALIES]

       

        # print("Unique test Normal Samples :test_normal",
        #       np.unique(test_normal))
        # print("Unique test Anomalous Samples :test_anomalies",
        #       np.unique(test_anomalies))
        label_normal = 1 * np.ones(len(test_normal))
        label_anomalies = 0 * np.zeros(len(test_anomalies))

        return [test_normal, label_normal, test_anomalies, label_anomalies]

    def get_USPS_TestingData(self):

        import tempfile
        import pickle

        with open(self.dataPath + 'usps_data.pkl', 'rb') as fp:
            loaded_data1 = pickle.load(fp, encoding='latin1')

        labels = loaded_data1['target']
        data = loaded_data1['data']

        # ## Scale the data
        # print(self.scaler.fit(data))
        StandardScaler(copy=True, with_mean=True, with_std=True)
        # data = self.scaler.transform(data)

        ## Select Ones
        k_ones = np.where(labels == 2)
        label_ones = labels[k_ones]
        data_ones = data[k_ones]

        k_sevens = np.where(labels == 8)
        label_sevens = labels[k_sevens]
        data_sevens = data[k_sevens]

        data_ones = data_ones[220:440]
        data_sevens = data_sevens[0:11]

        label_ones = 1 * np.ones(len(data_ones))
        label_sevens = np.zeros(len(data_sevens))

        return [data_ones, label_ones, data_sevens, label_sevens]

    def get_USPS_TestingData_With_Random_UniformNoise(self):

        import tempfile
        import pickle

        with open(self.dataPath + 'usps_data.pkl', 'rb') as fp:
            loaded_data1 = pickle.load(fp, encoding='latin1')

        labels = loaded_data1['target']
        data = loaded_data1['data']

        # ## Scale the data
        # print(self.scaler.fit(data))
        StandardScaler(copy=True, with_mean=True, with_std=True)
        # data = self.scaler.transform(data)

        ## Select Ones
        k_ones = np.where(labels == 2)
        label_ones = labels[k_ones]
        data_ones = data[k_ones]

        k_sevens = np.where(labels == 8)
        label_sevens = labels[k_sevens]
        data_sevens = data[k_sevens]

        data_ones = data_ones[220:440]
        data_sevens = np.random.uniform(0, 1, (len(data_ones), 256))

        label_ones = 1 * np.ones(len(data_ones))
        label_sevens = np.zeros(len(data_sevens))

        return [data_ones, label_ones, data_sevens, label_sevens]

    def get_USPS_TrainingData(self):

        import tempfile
        import pickle

        with open(self.dataPath + 'usps_data.pkl', 'rb') as fp:
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
        label_ones = 1 * np.ones(len(data_ones))

        return [data_ones, label_ones]

    def get_FAKE_Noise_TrainingData(self, X):

        data_noise = np.random.uniform(0, 1, (len(X), 256))
        label_noise = np.zeros(len(data_noise))
        return [data_noise, label_noise]
    
    def get_FAKE_Noise_LHC_TrainingData(self, X):

        data_noise = np.random.uniform(0, 1, (len(X), 1369))
        label_noise = np.zeros(len(data_noise))
        return [data_noise, label_noise]