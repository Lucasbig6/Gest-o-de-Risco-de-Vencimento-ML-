# 💊 Previsão Preditiva de Risco de Vencimento de Medicamentos (Machine Learning)

Sistema de Machine Learning para prever com antecedência quais lotes de medicamentos possuem **alto risco de vencer antes do uso**, reduzindo perdas financeiras e otimizando a logística de estoque.

---

## 📌 Sumário

- [Visão Geral](#-visão-geral)
- [Objetivos de Negócio](#-objetivos-de-negócio)
- [Arquitetura e Stack Tecnológico](#-arquitetura-e-stack-tecnológico)
- [Modelo de Machine Learning](#-modelo-de-machine-learning)
- [Principais Resultados](#-principais-resultados)
- [Demonstração da Aplicação](#-demonstração-da-aplicação)
- [Como Rodar o Projeto](#-como-rodar-o-projeto)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Próximos Passos](#-próximos-passos)
- [Licença](#-licença)

---

## 🌟 Visão Geral

Este projeto demonstra uma solução preditiva para um problema crítico em saúde e logística: o desperdício causado por medicamentos que vencem em estoque.

A proposta é um **classificador que prevê, com meses de antecedência**, se um lote possui **Alto Risco de Vencimento**, ajudando gestores a:

- Realocar lotes entre unidades
- Criar promoções ou acelerar consumo
- Planejar compras com mais precisão
- Reduzir perdas financeiras e aumentar ROI

---

## 🎯 Objetivos de Negócio

### O sistema entrega valor direto ao negócio por meio de:

- **Redução de Perdas (ROI):** diminui o volume perdido por vencimento.
- **Melhoria na Logística (FEFO):** reforça o método *First Expired, First Out*.
- **Suporte à Decisão:** transforma previsões complexas em ações práticas.

---

## 🛠️ Arquitetura e Stack Tecnológico

Este é um projeto **end-to-end**, cobrindo todas as etapas: dados → modelo → visualização.

### **Linguagens e Bibliotecas**

- **Python**
- **scikit-learn** — modelo de ML (Random Forest)
- **pandas** e **numpy** — tratamento de dados
- **matplotlib** e **seaborn** — visualizações e interpretabilidade
- **Streamlit** — interface interativa
- **joblib** — salvar e carregar o modelo

---

## 🤖 Modelo de Machine Learning

O modelo utilizado é um **Random Forest Classifier**, ideal para cenários com múltiplas features e interações não lineares.

| Tipo de Modelo | Algoritmo                | Objetivo                                               |
|----------------|---------------------------|--------------------------------------------------------|
| Classificação  | Random Forest Classifier | Prever risco binário: 0 = Baixo Risco / 1 = Alto Risco |

### **Variáveis Utilizadas**

- `Dias_Ate_Vencimento`
- `Estoque_Atual_unidades`
- `Taxa_Venda_Media_Dia`

Essas features foram escolhidas por representarem diretamente risco, giro e urgência do lote.

---

## 📊 Principais Resultados

A análise prioriza:

### ✔ Recall da Classe 1 (Alto Risco)  
Porque **não identificar um lote que vai vencer é o pior erro possível** (falso negativo).

A aplicação exibe:

- **Feature Importance** – importância de cada variável  
- **Matriz de Confusão** – erros críticos (especialmente classe 1)  
- **Probabilidades e insights individuais** (dependendo da versão)

---

## 🖥 Demonstração da Aplicação

O sistema possui uma interface desenvolvida em **Streamlit**, permitindo:

- Inserir dados manualmente  
- Ver previsão imediata  
- Interpretar a decisão do modelo  
- Explorar gráficos e estatísticas  

---

## 🚀 Como Rodar o Projeto

### **Pré-requisitos**

- Python **3.8+**
- Pip atualizado

---

### **1️⃣ Clonar o Repositório**

```bash
git clone https://docs.github.com/pt/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
cd nome-do-seu-projeto
```
### **2️⃣ Instalar Dependências**
```bash
pip install streamlit joblib pandas numpy scikit-learn matplotlib seaborn
```
### **3️⃣ Verificar se o modelo existe**

O arquivo esperado é:
```bash
modelo_risco_vencimento.joblib
```
Se não existir, rode o código de treinamento (posso gerar esse arquivo para você).

### **4️⃣ Executar o App**
```bash
streamlit run app.py
```

O navegador abrirá automaticamente em:

```bash
http://localhost:8501
```

### **📂 Estrutura do Projeto**
```bash
📁 projeto-risco-medicamentos
│
├── app.py                     # Interface Streamlit
├── modelo_risco_vencimento.joblib
├── treinar_modelo.py          # (Opcional) script para treinar o modelo
├── requirements.txt           # Dependências
└── README.md                  # Este arquivo
```

Posso gerar todos esses arquivos para você se quiser.

### **📄 Licença**
Este projeto é licenciado sob a MIT License — livre para uso pessoal e comercial.

### **✨ Autor**
Projeto desenvolvido por Lucas Araújo, focado em soluções de Data Science e IA para Saúde, Clima e Operações.
