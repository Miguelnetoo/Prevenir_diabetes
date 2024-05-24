from flask import Flask, request, render_template
from warnings import filterwarnings
import pickle

filterwarnings('ignore')

def import_model(): 
    # Abre o modelo treinado
    modelo = pickle.load(open('./modelo_logistico.pkl', 'rb'))
    return modelo

modelo = import_model()
app = Flask(__name__)

@app.route('/') 
def index():
    return render_template('forms_diabetes.html')

@app.route('/predict', methods=['POST']) 
def predict():
    # Obter os parâmetros do formulário
    parametros = [
        float(request.form['Pregnancies']), 
        float(request.form['Glucose']), 
        float(request.form['BloodPressure']), 
        float(request.form['SkinThickness']), 
        float(request.form['Insulin']), 
        float(request.form['BMI']), 
        float(request.form['DiabetesPedigreeFunction']), 
        float(request.form['Age'])
    ]
    
    # Fazer a predição
    resultado = modelo.predict([parametros])[0]
    
    # Interpretação do resultado
    if resultado == 0: 
        resultado = 'Não há chances de Diabetes'
    else:
        resultado = 'Há chances de Diabetes'

    return f'Seu resultado é: "{resultado}"!'

if __name__ == '__main__':
    app.run(
        debug=True,
        port=5000
    )
