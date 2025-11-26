import streamlit as st
import joblib
import pandas as pd
import numpy as np
# Importando Altair para gráficos declarativos e interativos
import altair as alt 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- Configuração Inicial ---
st.set_page_config(
    page_icon="💊",
    page_title="Gestão de Risco de Vencimento (ML)",
    layout="wide"
)

# --- Carregar o Modelo e Dados de Demonstração (Simulação) ---
try:
    # Tenta carregar o modelo salvo (necessita do arquivo .joblib)
    modelo = joblib.load('modelo_risco_vencimento.joblib')
except FileNotFoundError:
    st.error("Erro: O arquivo do modelo 'modelo_risco_vencimento.joblib' não foi encontrado. Execute o script de treinamento primeiro para salvar o modelo.")
    st.stop()

FEATURE_NAMES = ['Estoque_Atual_unidades', 'Dias_Ate_Vencimento', 'Taxa_Venda_Media_Dia']

# --- SIMULAÇÃO DE DADOS PARA O DASHBOARD ---
# Em um projeto real, esses dados seriam puxados de um banco de dados (ERP)
NUM_DADOS_MOCK = 5000
df_mock = pd.DataFrame({
    'Estoque_Atual_unidades': np.random.randint(100, 8000, NUM_DADOS_MOCK),
    'Dias_Ate_Vencimento': np.random.randint(1, 730, NUM_DADOS_MOCK),
    'Taxa_Venda_Media_Dia': np.random.randint(1, 70, NUM_DADOS_MOCK),
    'Custo_por_Unidade_R$': np.round(np.random.uniform(20, 700, NUM_DADOS_MOCK), 2)
})
# Lógica do Target (A mesma do treinamento)
df_mock['Risco_Vencimento'] = np.where(
    (df_mock['Dias_Ate_Vencimento'] < 180) & 
    (df_mock['Estoque_Atual_unidades'] > 30 * df_mock['Taxa_Venda_Media_Dia']),
    1, 
    0  
)
df_mock['Valor_Total_R$'] = df_mock['Estoque_Atual_unidades'] * df_mock['Custo_por_Unidade_R$']

# Fazer a previsão no dataset mock para o dashboard
df_mock['Previsao_Risco'] = modelo.predict(df_mock[FEATURE_NAMES])


# --- Função de Previsão de Lote Único ---
def prever_risco(estoque, dias, taxa_venda):
    """Recebe os inputs do usuário e retorna a previsão e a probabilidade."""
    dados_input = pd.DataFrame([[estoque, dias, taxa_venda]], columns=FEATURE_NAMES)
    previsao = modelo.predict(dados_input)[0]
    probabilidade = modelo.predict_proba(dados_input)[0][previsao]
    return previsao, probabilidade


# --- 1. SIDEBAR (Análise de Lote) ---
with st.sidebar:
    st.header("Análise Rápida de Lote")
    st.markdown("Insira os dados de um lote específico para verificar o risco de vencimento.")

    estoque_input = st.slider('Estoque Atual (Unidades)', min_value=0, max_value=10000, value=1500)
    dias_input = st.slider('Dias Até o Vencimento', min_value=0, max_value=730, value=120)
    taxa_venda_input = st.slider('Venda Média Diária (Unidades/Dia)', min_value=1, max_value=100, value=10)

    if st.button('Prever Risco para Lote'):
        risco, probabilidade = prever_risco(estoque_input, dias_input, taxa_venda_input)

        st.markdown("---")
        st.subheader("Resultado:")

        if risco == 1:
            st.error(f"### 🚨 RISCO ALTO DE PERDA")
            st.markdown(f"**Probabilidade:** `{probabilidade*100:.2f}%` de risco confirmado.")
            st.warning("AÇÃO: O lote tem mais estoque do que será vendido. Considere transferência imediata ou promoção.")
        else:
            st.success(f"### ✅ RISCO BAIXO")
            st.markdown(f"**Probabilidade:** `{probabilidade*100:.2f}%` de segurança.")
            st.info("AÇÃO: Monitoramento padrão. Estoque alinhado com o giro.")
        
        st.markdown("---")
        st.markdown("###### Detalhes do Lote:")
        st.markdown(f"- **Estoque Suficiente para:** `{estoque_input / taxa_venda_input:.1f}` dias")
        st.markdown(f"- **Tempo restante:** `{dias_input}` dias")

# --- 2. PAINEL PRINCIPAL (Métricas de Negócio) ---

st.title("💊 Painel de Controle Preditivo de Estoque Farmacêutico")
st.markdown("Visão executiva da saúde do estoque, focada em prevenir perdas financeiras por vencimento.")

risco_alto_df = df_mock[df_mock['Previsao_Risco'] == 1]
total_risco_alto = risco_alto_df.shape[0]
valor_total_em_risco = risco_alto_df['Valor_Total_R$'].sum()
media_dias_risco = risco_alto_df['Dias_Ate_Vencimento'].mean() if total_risco_alto > 0 else 0


col1, col2, col3 = st.columns(3)

with col1:
    # Removido border=True: st.metric não suporta borda diretamente
    st.metric(
        label="Total de Lotes Monitorados",
        value=f"{NUM_DADOS_MOCK:,}".replace(',', '.'),
        delta="Visão completa",
        border=True
    )

with col2:
    # Removido border=True: st.metric não suporta borda diretamente
    st.metric(
        label="Lotes Atualmente em ALTO RISCO",
        value=f"{total_risco_alto:,}".replace(',', '.'),
        delta=f"{(total_risco_alto / NUM_DADOS_MOCK * 100):.1f}% do total",
        delta_color="inverse",
        border=True
    )

with col3:
    # Formatação de moeda brasileira (R$)
    valor_formatado = f"R$ {valor_total_em_risco:,.2f}".replace(',', '_').replace('.', ',').replace('_', '.')
    # Removido border=True: st.metric não suporta borda diretamente
    st.metric(
        label="Valor Financeiro TOTAL em Risco",
        value=valor_formatado,
        delta=f"Média de {media_dias_risco:.0f} dias até o vencimento",
        delta_color="inverse",
        border=True
    )

st.markdown("---")

# --- 3. EXPLICABILIDADE SIMPLIFICADA (Feature Importance) ---
st.header("Entendendo a Decisão do Modelo (Explicabilidade)")

with st.container(border=True): # Container com borda para agrupar o gráfico e a explicação
    try:
        importances = modelo.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        
        # Criando nomes mais amigáveis para o gráfico
        friendly_names = {
            'Dias_Ate_Vencimento': 'Tempo Restante (Dias)',
            'Estoque_Atual_unidades': 'Volume em Estoque (Unid.)',
            'Taxa_Venda_Media_Dia': 'Giro de Vendas (Diário)'
        }
        
        # 1. Preparar dados para Altair
        importance_df = pd.DataFrame({
            # Mapeia nomes amigáveis e ordena
            'Fator de Risco': [friendly_names.get(name) for name in np.array(FEATURE_NAMES)[sorted_indices]],
            'Importância': importances[sorted_indices]
        })
        
        # 2. Criar a camada de barras base (Aumento de altura implementado abaixo)
        bar_chart = alt.Chart(importance_df).mark_bar().encode(
            # Y-axis ordenado pela Importância (descendente)
            y=alt.Y('Fator de Risco', sort='-x', title='Fator de Risco'),
            # X-axis para o valor da Importância
            x=alt.X('Importância', title='Nível de Importância para o Modelo'),
            # Cor dos bares
            color=alt.value('#00a68d'), 
            # Tooltip para interatividade
            tooltip=['Fator de Risco', alt.Tooltip('Importância', format='.2f')]
        )
        
        # 3. Criar a camada de texto (Rótulos nas barras)
        text_layer = bar_chart.mark_text(
            align='left', 
            baseline='middle', 
            dx=3 # Desloca o texto um pouco para a direita da barra
        ).encode(
            # Usa o valor de Importância como rótulo, formatado
            text=alt.Text('Importância', format='.2f'), 
            # Define a cor do texto para melhor contraste
            color=alt.value('black')
        )
        
        # 4. Combinar as camadas, definir a altura e exibir
        final_chart = (bar_chart + text_layer).properties(
            title='Quais Fatores Mais Determinam o Risco de Perda?',
            height=350 # Aumentando a altura do gráfico
        ).interactive() # Permite zoom e pan

        # Exibir o gráfico no Streamlit
        st.altair_chart(final_chart, use_container_width=True)
        
        st.markdown("""
        Este gráfico mostra o foco do modelo. Normalmente, o **Tempo Restante** é o mais crítico, mas se o **Volume em Estoque** for muito alto, a influência dele se iguala. Isso é a chave para a gestão proativa.
        """)
        
    except Exception as e:
        # A Matplotlib/Seaborn já não é usada aqui, mas as exceções são tratadas.
        st.error(f"Não foi possível gerar o gráfico de Importância das Features.") 

st.markdown("---")

# --- 4. ANÁLISE AVANÇADA (Oculta para o usuário final) ---
with st.expander("🛠️ Análise Avançada e Validação do Modelo (Para a equipe de DS)"):
    
    st.subheader("Validação de Desempenho do Modelo")
    
    # Usamos os dados de teste simulados
    X_teste_simulado = df_mock[FEATURE_NAMES]
    y_teste_simulado_real = df_mock['Risco_Vencimento']
    y_teste_simulado_predito = df_mock['Previsao_Risco']
    
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Matriz de Confusão (Dados de Demonstração)")
        # Este bloco AINDA USA Matplotlib/Seaborn para a Matriz de Confusão
        cm = confusion_matrix(y_teste_simulado_real, y_teste_simulado_predito)
        cm_df = pd.DataFrame(cm, 
                             index = ['Real Baixo Risco', 'Real Alto Risco'], 
                             columns = ['Previsto Baixo Risco', 'Previsto Alto Risco'])
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_df, annot=True, fmt='g', cmap='viridis', cbar=False, ax=ax_cm)
        ax_cm.set_title('Matriz de Confusão')
        st.pyplot(fig_cm)
        
        st.markdown(f"""
        - **Falsos Negativos (Risco Perdido):** {cm[1, 0]}
        """)

    with col2:
        st.caption("Relatório de Classificação Detalhado")
        report = classification_report(y_teste_simulado_real, y_teste_simulado_predito, target_names=['Baixo Risco', 'Alto Risco'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
        
        st.markdown("""
        - **Foco:** O `Recall` de 'Alto Risco' é a métrica mais importante, medindo nossa capacidade de CAPTURAR todos os lotes problemáticos.
        """)

st.markdown("---")
st.markdown("###### Desenvolvido por Lucas Pablo - Cientista de Dados | 2025")