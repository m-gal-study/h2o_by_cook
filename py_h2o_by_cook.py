# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#h2o.shutdown()

####################################################################################################
######################## S T A R T #################################################################
import h2o
h2o.init()


#### BEG: TOPIC 1
data = h2o.import_file("C:/Users/mikhail.galkin/Documents/RProjects/r201807-Tutorial h2o/iris_wheader.csv")
data.names
y = 'class'
x = data.names
x.remove(y)
x
train, test = data.split_frame([0.8], seed = 99)
m = h2o.estimators.deeplearning.H2ODeepLearningEstimator()
m.train(x, y, train)
p = m.predict(test)


m.mse()
m.confusion_matrix(train)
p.as_data_frame()
(p["predict"] == test["class"]).mean()
p["predict"].cbind(test["class"]).as_data_frame()
m.model_performance(test)
#### END: TOPIC 1 



#### BEG: TOPIC 2
patients = {
        'heigth' : [188, 157, 157],
        'age' : [29, 33, 65],
        'risk' : ['A', 'B', 'B']
        }
df = h2o.H2OFrame(patients)
df.types



patients = {
        'heigth' : [188, 157, 175.1],
        'age' : [29, 33, 65],
        'risk' : ['A', 'B', 'B']
        }
df1 = h2o.H2OFrame.from_python(
        patients,
        column_types = [None, 'enum', None],
        destination_frame = 'patients'
        )
df1.types
df1.frame_id



import pandas as pd
patients = pd.DataFrame({
        'heigth' : [188, 157, 175.1],
        'age' : [29, 33, 65],
        'risk' : ['A', 'B', 'B']
        })
df2 = h2o.H2OFrame(patients)
df2.types
df2.frame_id



h2o.shutdown()
import h2o
h2o.init()


data = h2o.import_file("C:/Users/mikhail.galkin/Documents/RProjects/r201807-Tutorial h2o/iris_wheader.csv")
data.frame_id
data = data[:, 1:]
data.frame_id

data = h2o.assign(data, 'iris')
data.frame_id

h2o.ls()
h2o.remove('iris_wheader3.hex')

data.describe()

data.as_data_frame()
data['petal_len'] = data['petal_len'] * 1.2
data.as_data_frame()
data.frame_id

data['ratio'] = data['petal_wid'] / data['sepal_wid']
data.as_data_frame()

data['petal_len'].sd()

data['ratio'].cor(data['petal_len'])

data['islong'] = (data['petal_len'] > data['petal_len'].mean()[0]).ifelse(1, 0)
data.as_data_frame()

data['species'] = data['class'].ascharacter().gsub('Iris-', '')
data.as_data_frame()
data.frame_id

data.group_by('class').count().mean('petal_len').sum('islong').frame
data['petal_len'].hist()




