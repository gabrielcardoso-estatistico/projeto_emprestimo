import numpy as np
import os
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Verificar estrutura
print("=" * 50)
print("VERIFICANDO ESTRUTURA DO PROJETO")
print("=" * 50)
print(f"Diretório atual: {os.getcwd()}")
print(f"Root path do app: {app.root_path}")
print(f"Pasta templates: {app.template_folder}")
print(f"Templates existe: {os.path.exists(app.template_folder)}")

# Listar arquivos na pasta templates
if os.path.exists(app.template_folder):
    print("\nArquivos em templates/:")
    for file in os.listdir(app.template_folder):
        print(f"  - {file}")
else:
    print(f"\n⚠️ ATENÇÃO: Pasta templates não encontrada!")
    print(f"Criando pasta templates...")
    os.makedirs(app.template_folder, exist_ok=True)

# Carregar o modelo
try:
    model_path = os.path.join(app.root_path, 'model', 'modelo_regressao_logistica.pkl')
    print(f"\nTentando carregar modelo de: {model_path}")
    print(f"Arquivo do modelo existe: {os.path.exists(model_path)}")
    
    model = joblib.load(model_path)
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    model = None

@app.route('/')
def home():
    print(f"\n[ROTA /] Renderizando index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(f"\n[ROTA /predict] Recebendo requisição...")
    
    if model is None:
        print("⚠️ Modelo não carregado!")
        return render_template('index.html', 
                             resultado="Erro: Modelo não carregado",
                             status="erro")

    try:
        # Coletar dados do formulário
        print("Coletando dados do formulário...")
        dados = [
            float(request.form.get('sexo', 0)),        # 1=Masculino, 0=Feminino
            float(request.form.get('casado', 0)),      # 1=Sim, 0=Não
            float(request.form.get('dependentes', 0)),
            float(request.form.get('educacao', 0)),    # 1=Graduado, 0=Não Graduado
            float(request.form.get('conta_propria', 0)), # 1=Sim, 0=Não
            float(request.form.get('rendimento', 0)),
            float(request.form.get('valor_emprestimo', 0))
        ]
        print(f"Dados coletados: {dados}")
    except Exception as e:
        print(f"❌ Erro nos dados: {e}")
        return render_template('index.html', 
                             resultado="Erro: Dados inválidos",
                             status="erro")

    # Fazer predição
    X = np.array([dados])
    
    try:
        print("Fazendo predição...")
        # Resultado da predição
        predicao = int(model.predict(X)[0])
        print(f"Predição: {predicao}")
        
        # Probabilidades
        probabilidades = model.predict_proba(X)[0]
        print(f"Probabilidades: {probabilidades}")
        
        # Calcular probabilidade de aprovação
        prob_aprovado = round(probabilidades[1] * 100, 2)
        
        # Definir resultado
        if predicao == 1:
            resultado = "✅ EMPRÉSTIMO APROVADO"
            status = "aprovado"
        else:
            resultado = "❌ EMPRÉSTIMO NEGADO"
            status = "negado"
        
        print(f"Resultado: {resultado}")
        print(f"Probabilidade: {prob_aprovado}%")
        
        return render_template('index.html',
                             resultado=resultado,
                             status=status,
                             probabilidade=prob_aprovado)
    
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return render_template('index.html',
                             resultado=f"Erro na predição: {str(e)}",
                             status="erro")

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("INICIANDO SERVIDOR FLASK")
    print("=" * 50)
    
    # Verificar se index.html existe
    index_path = os.path.join(app.template_folder, 'index.html')
    print(f"\nVerificando template index.html:")
    print(f"Caminho: {index_path}")
    print(f"Existe: {os.path.exists(index_path)}")
    
    if not os.path.exists(index_path):
        print("\n⚠️ ATENÇÃO: index.html não encontrado!")
        print("Criando template básico...")
        
        # Criar template básico
        template_basico = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sistema de Análise de Crédito</title>
            <style>
                body { font-family: Arial; padding: 20px; }
                .form-container { max-width: 500px; margin: auto; }
                input, select { width: 100%; padding: 8px; margin: 5px 0; }
                button { background: #007bff; color: white; padding: 10px; border: none; }
                .result { padding: 20px; margin: 20px 0; border-radius: 5px; }
                .aprovado { background: green; color: white; }
                .negado { background: red; color: white; }
                .erro { background: orange; color: white; }
            </style>
        </head>
        <body>
            <div class="form-container">
                <h1>Análise de Crédito</h1>
                <form action="/predict" method="POST">
                    Sexo: 
                    <select name="sexo"><option value="1">Masculino</option><option value="0">Feminino</option></select><br>
                    Casado: 
                    <select name="casado"><option value="1">Sim</option><option value="0">Não</option></select><br>
                    Dependentes: <input type="number" name="dependentes" value="0"><br>
                    Graduação: 
                    <select name="educacao"><option value="1">Graduado</option><option value="0">Não Graduado</option></select><br>
                    Conta Própria: 
                    <select name="conta_propria"><option value="1">Sim</option><option value="0">Não</option></select><br>
                    Rendimento: <input type="number" name="rendimento" step="0.01"><br>
                    Valor Empréstimo: <input type="number" name="valor_emprestimo" step="0.01"><br>
                    <button type="submit">ANALISAR</button>
                </form>
                
                {% if resultado %}
                <div class="result {{ status }}">
                    <h2>{{ resultado }}</h2>
                    {% if probabilidade %}
                    <p>Probabilidade: {{ probabilidade }}%</p>
                    {% endif %}
                </div>
                {% endif %}
            </div>
        </body>
        </html>
        """
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(template_basico)
        print("✅ Template criado com sucesso!")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Servidor iniciando na porta {port}")
    print(f"📁 URL: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)