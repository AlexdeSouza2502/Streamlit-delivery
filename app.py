import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Caminho da imagem
image_path = "/workspaces/Streamlit-delivery/img/delivery img.jpg"

# 1. Carregar os Dados
df = None  # Inicializa df com None antes do try

try:
    df = pd.read_csv("estabelecimentos.csv", encoding='utf-8')  # Primeiro com UTF-8
except UnicodeDecodeError:
    df = pd.read_csv("estabelecimentos.csv", encoding='latin1')  # Se falhar, tente Latin1
except FileNotFoundError:
    st.error(
        "Arquivo 'estabelecimentos.csv' não encontrado. Certifique-se de que ele está no mesmo diretório do script."
    )
    df = None  # Define df como None em caso de erro
    st.stop()

if df is None:  # Verifica se houve erro ao carregar o arquivo
    st.stop()  # Para a execução do script

# 2. Pré-Processamento

# **Adaptar Colunas**
TARGET_COLUMN = 'faz_delivery'  # Queremos prever se o estabelecimento faz delivery
FEATURES = ['aceita_cupom', 'avaliacao', 'taxa_entrega']  # Features que usaremos para prever

# Converter colunas booleanas para numéricas (0 e 1)
BOOLEAN_FEATURES = ['indisponivel', 'aceita_cupom', 'faz_delivery', 'faz_retirada', 'tem_promocao']

# Adicionar verificação para df antes de acessar df.columns
if df is not None:
    # Converter booleanas e preencher NaN
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0})  # Converte True/False para 1/0
            df[col] = df[col].fillna(0)  # Preenche valores faltantes com 0 (assumindo False)
        else:
            st.warning(f"Coluna booleana '{col}' não encontrada. Ignorando.")

    # Tratar feature 'avaliacao'
    if 'avaliacao' in df.columns:
        # Extrair o valor numérico de 'estrelas' (se o formato for '{"estrelas":valor,...}')
        df['avaliacao'] = df['avaliacao'].str.extract(r'"estrelas":([\d\.]+)')
        df['avaliacao'] = pd.to_numeric(df['avaliacao'], errors='coerce')  # Converte para numérico
        df['avaliacao'] = df['avaliacao'].fillna(df['avaliacao'].mean())  # Preenche NaN com a média
    else:
        st.warning("Coluna 'avaliacao' não encontrada. Ignorando.")
        FEATURES.remove('avaliacao') # Remove a feature se ela não existe

    # Tratar feature 'taxa_entrega'
    if 'taxa_entrega' in df.columns:
        # Extrair o valor da taxa de entrega (se o formato for '{"min":valor,"max":valor,"valor":valor}')
        df['taxa_min'] = df['taxa_entrega'].str.extract(r'"min":([\d\.]+)')
        df['taxa_max'] = df['taxa_entrega'].str.extract(r'"max":([\d\.]+)')
        df['taxa_valor'] = df['taxa_entrega'].str.extract(r'"valor":([\d\.]+)')

        # Convertendo para numeric e preenchendo com a média
        for col in ['taxa_min', 'taxa_max', 'taxa_valor']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())

        # Manter apenas a taxa
        df['taxa_entrega'] = df[['taxa_min', 'taxa_max', 'taxa_valor']].mean(axis=1)

    else:
        st.warning("Coluna 'taxa_entrega' não encontrada. Ignorando.")
        FEATURES.remove('taxa_entrega') # Remove a feature se ela não existe

    # Remover linhas com valores faltantes na coluna alvo
    df = df.dropna(subset=[TARGET_COLUMN])

    # Garantir que a coluna alvo seja numérica
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Remover colunas que não existem ou tem valores faltantes
    FEATURES = [f for f in FEATURES if f in df.columns]

    # Remover linhas onde qualquer feature tenha NaN APÓS a conversão
    df = df.dropna(subset=FEATURES)  # Remove qualquer linha que tenha NaN nas features

    # Remover duplicatas
    df = df.drop_duplicates()

    # Ajustar o tamanho do teste se necessário
    test_size = 0.2
    if len(df) < 50:  # Ajustar o tamanho do teste para conjuntos de dados pequenos
        test_size = 0.05

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET_COLUMN], test_size=test_size, random_state=42
    )

    # 3. Treinar o Modelo (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 4. Avaliar o Modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 5. Interface Streamlit
    col1, col2 = st.columns([1, 3])  # Divide a linha em duas colunas

    with col1:
        try:
            st.image(image_path, width=200)  # Aumenta a largura para 200 pixels
        except FileNotFoundError:
            st.error("Imagem não encontrada. Verifique se o caminho está correto.")

    with col2:
        st.title("Priorização de Estabelecimentos para Delivery")

    st.markdown("Análise de potenciais parceiros com base em dados do mercado.")

    # **6. Sidebar Interativo**
    with st.sidebar:
        st.header("Filtros")
        # Remove caracteres extras das cidades
        df["cidade"] = df["cidade"].str.replace(r'[\[\]"\']', '', regex=True)
        cidades_unicas = sorted(list(df["cidade"].unique()))
        cidade_filtro = st.selectbox("Selecione a Cidade", options=cidades_unicas, index=0)

        # Limpeza das categorias: remover "[]" e aspas
        df["categorias"] = df["categorias"].str.replace(r'[\[\]"\']', '', regex=True)
        # Separa as categorias se houver múltiplas (ex: "Bar, Tapiocaria")
        categorias_unicas = []
        for cats in df["categorias"].unique():
            categorias_unicas.extend(cats.split(", "))  # Separa por ", "
        categorias_unicas = sorted(list(set(categorias_unicas)))  # Remove duplicatas e ordena

        categoria_filtro = st.multiselect("Categorias", options=categorias_unicas, default=categorias_unicas)
        st.markdown("""---""") # Linha divisória
        st.markdown("Desenvolvido por [8ºSI](https://github.com/AlexdeSouza2502)")

    # 7. Previsões e Scores

    # Calcular o score de potencial
    df['score_potencial'] = model.predict_proba(df[FEATURES])[:, 1]

    # Ordenar por score
    df_ordenado = df.sort_values(by='score_potencial', ascending=False)

    # **8. Filtrar os dados**

    # Filtrar os dados com as categorias tratadas
    df_filtrado = df_ordenado[df_ordenado["cidade"] == cidade_filtro]
    df_filtrado = df_filtrado[df_filtrado["categorias"].apply(lambda x: any(cat in x for cat in categoria_filtro))]

    # 9. Melhorar a Exibição da Tabela
    def highlight_score(val):
        color = 'lightgreen' if val > 0.7 else 'lightcoral'
        return f'background-color: {color}'

    st.dataframe(df_filtrado.style.applymap(highlight_score, subset=['score_potencial']))

    # 10. Métricas de Avaliação
    st.subheader("Métricas do Modelo")
    st.metric("Acurácia do Modelo", f"{accuracy:.2f}")

    # 11. Visualizações (Exemplos)
    st.subheader("Visualizações")

    # Distribuição de Scores
    fig_scores = px.histogram(df_filtrado, x='score_potencial', title="Distribuição dos Scores de Potencial")
    st.plotly_chart(fig_scores, use_container_width=True)

    # Relação Avaliação vs Score
    fig_rating_score = px.scatter(
        df_filtrado,
        x='avaliacao',
        y='score_potencial',
        title="Relação entre Avaliação e Score de Potencial",
        hover_data=['estabelecimento'], # Mostrar o nome do estabelecimento
    )  # Mostrar o nome ao passar o mouse
    st.plotly_chart(fig_rating_score, use_container_width=True)

     # 12. Adicionar um Gráfico de Barras para as Categorias
    fig_categorias = px.bar(df_filtrado.groupby("categorias")["score_potencial"].mean().reset_index(),
                            x="categorias", y="score_potencial", title="Média de Score por Categoria")
    st.plotly_chart(fig_categorias)
else:
    st.error("Ocorreu um erro ao carregar o arquivo CSV. Verifique se ele existe e está no formato correto.")
