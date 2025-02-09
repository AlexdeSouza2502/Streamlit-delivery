import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Carregar os Dados
try:
    df = pd.read_csv("estabelecimentos.csv", encoding='utf-8') # Primeiro com UTF-8
except UnicodeDecodeError:
    df = pd.read_csv("estabelecimentos.csv", encoding='latin1') # Se falhar, tente Latin1
except FileNotFoundError:
    st.error("Arquivo 'estabelecimentos.csv' não encontrado. Certifique-se de que ele está no mesmo diretório do script.")
    st.stop()

# 2. Pré-Processamento

# **Adaptar Colunas**
TARGET_COLUMN = 'faz_delivery'  # Queremos prever se o estabelecimento faz delivery
FEATURES = ['aceita_cupom', 'avaliacao', 'taxa_entrega']  # Features que usaremos para prever

# Converter colunas booleanas para numéricas (0 e 1)
# Precisamos saber quais colunas são booleanas no seu CSV
BOOLEAN_FEATURES = ['indisponivel', 'aceita_cupom', 'faz_delivery', 'faz_retirada', 'tem_promocao']
for col in BOOLEAN_FEATURES:
    if col in df.columns:
        df[col] = df[col].map({True: 1, False: 0})  # Converte True/False para 1/0
        df[col] = df[col].fillna(0)  # Preenche valores faltantes com 0 (assumindo False)

# Tratar valores faltantes (NaN) nas features numéricas
for col in FEATURES:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())  # Preenche com a média

# Remover linhas com valores faltantes na coluna alvo
df = df.dropna(subset=[TARGET_COLUMN])

# Garantir que a coluna alvo seja numérica
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

# Remover colunas que não existem ou tem valores faltantes
FEATURES = [f for f in FEATURES if f in df.columns]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET_COLUMN], test_size=0.2, random_state=42)

# 3. Treinar o Modelo (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Avaliar o Modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 5. Interface Streamlit
st.title("Priorização de Estabelecimentos para Delivery")
st.markdown("Análise de potenciais parceiros com base em dados do mercado.")

# 6. Previsões e Scores
st.subheader("Score de Potencial dos Estabelecimentos")

# Calcular o score de potencial
df['score_potencial'] = model.predict_proba(df[FEATURES])[:, 1]

# Ordenar por score
df_ordenado = df.sort_values(by='score_potencial', ascending=False)

# Exibir a tabela ordenada
st.dataframe(df_ordenado[['nome_fantasia', 'tipo_culinaria', 'cidade', 'avaliacao', 'score_potencial']])

# 7. Métricas de Avaliação
st.subheader("Métricas do Modelo")
st.metric("Acurácia do Modelo", f"{accuracy:.2f}")

# 8. Visualizações (Exemplos)
st.subheader("Visualizações")

# Distribuição de Scores
fig_scores = px.histogram(df_ordenado, x='score_potencial', title="Distribuição dos Scores de Potencial")
st.plotly_chart(fig_scores, use_container_width=True)

# Relação Avaliação vs Score
fig_rating_score = px.scatter(df_ordenado, x='avaliacao', y='score_potencial',
                                 title="Relação entre Avaliação e Score de Potencial",
                                 hover_data=['nome_fantasia'])  # Mostrar o nome ao passar o mouse
st.plotly_chart(fig_rating_score, use_container_width=True)
