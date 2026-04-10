import pickle
def detact(x):
    with open("model.pkl","rb") as model:
        model_data = pickle.load(model)
    
    with open("vectorizer.pkl","rb") as model:
        vector_data = pickle.load(model)
        
    v= vector_data.transform([x])
    print(model_data.predict(v))
        
        
detact("modei attck on india")