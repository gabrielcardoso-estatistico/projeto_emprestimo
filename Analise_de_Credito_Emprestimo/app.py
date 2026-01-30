import numpy as np
import os
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Carregar o modelo
try:
    model_path = os.path.join(app.root_path, 'model', 'modelo_regressao_logistica.pkl')
    model = joblib.load(model_path)
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                             resultado="Erro: Modelo não carregado",
                             status="erro")

    try:
        # Coletar dados do formulário
        dados = [
            float(request.form.get('sexo', 0)),        # 1=Masculino, 0=Feminino
            float(request.form.get('casado', 0)),      # 1=Sim, 0=Não
            float(request.form.get('dependentes', 0)),
            float(request.form.get('educacao', 0)),    # 1=Graduado, 0=Não Graduado
            float(request.form.get('conta_propria', 0)), # 1=Sim, 0=Não
            float(request.form.get('rendimento', 0)),
            float(request.form.get('valor_emprestimo', 0))
        ]
    except:
        return render_template('index.html', 
                             resultado="Erro: Dados inválidos",
                             status="erro")

    # Fazer predição
    X = np.array([dados])
    
    try:
        # Resultado da predição
        predicao = int(model.predict(X)[0])
        
        # Probabilidades
        probabilidades = model.predict_proba(X)[0]
        
        # Calcular probabilidade de aprovação
        if predicao == 1:
            prob_aprovado = round(probabilidades[1] * 100, 2)
        else:
            prob_aprovado = round(probabilidades[0] * 100, 2)
        
        # Definir resultado
        if predicao == 1:
            resultado = "✅ EMPRÉSTIMO APROVADO"
            status = "aprovado"
        else:
            resultado = "❌ EMPRÉSTIMO NEGADO"
            status = "negado"
        
        return render_template('index.html',
                             resultado=resultado,
                             status=status,
                             probabilidade=prob_aprovado)
    
    except Exception as e:
        return render_template('template.html',
                             resultado=f"Erro na predição: {str(e)}",
                             status="erro")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)