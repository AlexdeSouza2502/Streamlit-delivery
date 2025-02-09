# 📊 Priorização de Estabelecimentos para Delivery

## 📌 Sobre o Projeto
Este projeto utiliza **Machine Learning** para analisar e priorizar estabelecimentos com potencial para delivery. Ele conta com um modelo de **Random Forest** para prever a probabilidade de um estabelecimento oferecer serviço de delivery com base em dados históricos. O aplicativo foi desenvolvido com **Streamlit** e utiliza **Plotly** para visualizações interativas.

## 🚀 Funcionalidades
- **Carregamento Dinâmico de Dados:** Leitura automática do arquivo CSV `estabelecimentos.csv`.
- **Pré-Processamento Inteligente:** Conversão de colunas booleanas, tratamento de valores nulos e normalização de variáveis categóricas.
- **Treinamento de Modelo:** Implementação de um **Random Forest Classifier** para prever se um estabelecimento faz delivery.
- **Métricas de Avaliação:** Exibição da **acurácia do modelo** e métricas de predição.
- **Filtros Interativos:** Selecione cidades e categorias de estabelecimentos para análise.
- **Visualizações Dinâmicas:** Histogramas, gráficos de dispersão e gráficos de barras para análise dos dados.

## 🛠 Tecnologias Utilizadas
- **Python 3.12**
- **Streamlit** (Interface interativa)
- **Pandas** (Manipulação de dados)
- **Plotly** (Gráficos interativos)
- **Scikit-learn** (Treinamento de modelo ML)

## 📂 Estrutura do Projeto
```
📦 Streamlit-Delivery
 ┣ 📂 img
 ┃ ┗ 📜 delivery img.jpg  # Imagem utilizada na interface
 ┣ 📜 estabelecimentos.csv  # Arquivo de dados (deve estar no diretório raiz)
 ┣ 📜 app.py  # Código principal do Streamlit
 ┣ 📜 requirements.txt  # Dependências do projeto
 ┗ 📜 README.md  # Documentação do projeto
```

## ▶️ Como Executar
### 1️⃣ Instalar Dependências
Certifique-se de ter o **Python** instalado. Em seguida, instale as dependências com:
```bash
pip install -r requirements.txt
```

### 2️⃣ Executar o Aplicativo
Após instalar as dependências, execute o Streamlit:
```bash
streamlit run app.py
```

## 📊 Exemplos de Uso
1. Selecione uma **cidade** e **categorias** na barra lateral.
2. Visualize a **lista priorizada** de estabelecimentos para delivery.
3. Analise métricas e gráficos interativos.

## 🛠 Problemas Comuns
- **Arquivo CSV não encontrado** → Certifique-se de que `estabelecimentos.csv` está no diretório correto.
- **Imagem não carregando** → Verifique o caminho do arquivo `delivery img.jpg`.
- **Erro de dependências** → Atualize as bibliotecas executando:
  ```bash
  pip install --upgrade -r requirements.txt
  ```

## 📌 Contato
Desenvolvido por **8ºSI**. Para mais detalhes, acesse meu GitHub:
[AlexdeSouza2502](https://github.com/AlexdeSouza2502)

