import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("Admission_Predict.csv")

# setting x and y
x = df.iloc[:, 1:8] #without Chance of Admit
y = df.iloc[:, -1] #only Chance of Admit

reg_model = LinearRegression()

reg_model.fit(x,y)

# saving the model to the current dir
# pickle serializes objects so that they can be saved to a file
pickle.dump(reg_model, open('model.pkl','wb'))

'''
THIS WORKED!
# Loading the model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([x.iloc[0, :]]))
'''