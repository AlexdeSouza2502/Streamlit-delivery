import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Carregar os Dados
try:
    df = pd.read_csv("estabelecimentos.csv", encoding='utf-8') # Tente com utf-8 primeiro
except UnicodeDecodeError:
    df = pd.read_csv("estabelecimentos.csv", encoding='latin1') # Se falhar, tente latin1

# 2. Configurar o Streamlit
st.title("Análise de Estabelecimentos")

# 3. Criar Filtros (adapte conforme as colunas do seu CSV)
# Seletor de Estado
estados_unicos = df['estado'].unique()
estado_selecionado = st.selectbox("Selecione o Estado:", estados_unicos)
df_filtrado = df[df['estado'] == estado_selecionado]

# Seletor de Tipo de Estabelecimento
tipos_unicos = df['tipo_estabelecimento'].unique() # Pega as opções para o filtro
tipo_selecionado = st.selectbox("Selecione o Tipo de Estabelecimento:", tipos_unicos)
df_filtrado = df_filtrado[df_filtrado['tipo_estabelecimento'] == tipo_selecionado]

# 4. Criar Visualizações

# Distribuição de Tipos de Estabelecimento (Gráfico de Barras)
st.subheader("Distribuição de Tipos de Estabelecimento")
distribuicao_tipos = df_filtrado['tipo_estabelecimento'].value_counts().reset_index()
distribuicao_tipos.columns = ['tipo_estabelecimento', 'quantidade']
fig_tipos = px.bar(distribuicao_tipos, x='tipo_estabelecimento', y='quantidade',
                    title=f"Distribuição de Tipos em {estado_selecionado}")
st.plotly_chart(fig_tipos, use_container_width=True)

# Distribuição de Estabelecimentos por Cidade (Gráfico de Barras)
st.subheader("Distribuição de Estabelecimentos por Cidade")
distribuicao_cidades = df_filtrado['cidade'].value_counts().nlargest(10).reset_index() # Top 10 cidades
distribuicao_cidades.columns = ['cidade', 'quantidade']
fig_cidades = px.bar(distribuicao_cidades, x='cidade', y='quantidade',
                    title=f"Top 10 Cidades em {estado_selecionado}")
st.plotly_chart(fig_cidades, use_container_width=True)
