import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Função para baixar o dataset se não existir
def download_dataset(url, filename):
    if not os.path.exists(filename):
        print(f"Baixando {filename}...")
        try:
            import requests
            response = requests.get(url)
            response.raise_for_status() # Levanta um erro para bad status codes
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"{filename} baixado com sucesso!")
        except Exception as e:
            print(f"Erro ao baixar {filename}: {e}")
            print("Por favor, baixe o arquivo manualmente e coloque-o na mesma pasta do notebook.")
    else:
        print(f"{filename} já existe. Pulando download.")

def analyze_missing_data(df):
    """Analisa e visualiza os dados faltantes"""
    print("=" * 60)
    print("ANÁLISE DE DADOS FALTANTES")
    print("=" * 60)

    # Análise por coluna
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100

    missing_info = pd.DataFrame({
        'Coluna': missing_data.index,
        'Valores Faltantes': missing_data.values,
        'Percentual (%)': missing_percent.values
    }).sort_values('Valores Faltantes', ascending=False)

    print("\nDados faltantes por coluna:")
    print(missing_info[missing_info['Valores Faltantes'] > 0].to_string(index=False))

    # Análise específica para 'stalk-root' por classe
    if 'stalk-root' in df.columns and 'class' in df.columns:
        print("\n" + "=" * 40)
        print("ANÁLISE DETALHADA - stalk-root")
        print("=" * 40)

        stalk_root_analysis = df.groupby(['class', 'stalk-root']).size().unstack(fill_value=0)
        print("\nDistribuição de 'stalk-root' por classe:")
        print(stalk_root_analysis)

        # Valores faltantes por classe
        missing_by_class = df[df['stalk-root'].isna()]['class'].value_counts()
        print(f"\nValores faltantes em 'stalk-root' por classe:")
        for cls, count in missing_by_class.items():
            print(f"  Classe '{cls}': {count} instâncias ({count/len(df[df['stalk-root'].isna()])*100:.1f}%)")

    return missing_info

def conservative_preprocessing():
    """Pré-processamento conservador para dados de cogumelos"""

    print("🚀 INICIANDO PRÉ-PROCESSAMENTO CONSERVADOR")
    print("=" * 60)

    # 1. DEFINIÇÃO DAS COLUNAS
    colunas = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
        'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
        'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
        'population', 'habitat'
    ]

    # Definir URL e nome do arquivo do dataset
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    dataset_filename = 'agaricus-lepiota.data'

    # Baixar o dataset se não existir
    download_dataset(dataset_url, dataset_filename)

    # 2. CARREGAMENTO DOS DADOS
    df = pd.read_csv(dataset_filename, header=None, names=colunas, na_values='?')

    print(f"📊 Dataset original carregado: {df.shape[0]} linhas e {df.shape[1]} colunas.")

    # 3. ANÁLISE DETALHADA DOS DADOS FALTANTES
    missing_info = analyze_missing_data(df)

    # 4. ABORDAGEM CONSERVADORA: REMOÇÃO DE INSTÂNCIAS INCOMPLETAS
    print("\n" + "=" * 60)
    print("APLICAÇÃO DA ABORDAGEM CONSERVADORA")
    print("=" * 60)

    df_clean = df.dropna()
    # Adicionar esta linha para resetar o índice após dropar os NaNs
    df_clean = df_clean.reset_index(drop=True)

    print(f"✅ Dataset após remoção de instâncias com dados faltantes:")
    print(f"   - Linhas originais: {len(df)}")
    print(f"   - Linhas mantidas: {len(df_clean)}")
    print(f"   - Redução: {len(df) - len(df_clean)} linhas removidas ({((len(df) - len(df_clean))/len(df))*100:.2f}%)")
    print(f"   - Dataset final: {df_clean.shape[0]} linhas × {df_clean.shape[1]} colunas")

    # 5. VERIFICAÇÃO DO BALANCEAMENTO APÓS LIMPEZA
    print("\n" + "=" * 40)
    print("VERIFICAÇÃO DO BALANCEAMENTO")
    print("=" * 40)

    class_distribution = df_clean['class'].value_counts()
    print("Distribuição das classes no dataset limpo:")
    for cls, count in class_distribution.items():
        percent = (count / len(df_clean)) * 100
        cls_name = "Comestível (e)" if cls == 'e' else "Venenoso (p)"
        print(f"  {cls_name}: {count} instâncias ({percent:.2f}%)")

    # 6. PRÉ-PROCESSAMENTO PARA MODELAGEM
    print("\n" + "=" * 60)
    print("PRÉ-PROCESSAMENTO FINAL")
    print("=" * 60)

    # Separar features e target
    X = df_clean.drop('class', axis=1)
    y = df_clean['class']

    # Codificação do target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # Mapeamento: 0 = edible (comestível), 1 = poisonous (venenoso)
    print(f"🔢 Target codificado: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # One-Hot Encoding para features categóricas
    X_encoded = pd.get_dummies(X)

    print(f"📈 Dataset final para modelagem:")
    print(f"   - Shape: {X_encoded.shape[0]} amostras, {X_encoded.shape[1]} features")
    print(f"   - Features após one-hot encoding: {X_encoded.shape[1]} colunas")

    # 7. SALVAR DATASET PROCESSADO
    df_final = pd.concat([pd.DataFrame(y_encoded, columns=['target']), X_encoded], axis=1)
    df_final.to_csv('mushroom_processed_conservative.csv', index=False)

    print("\n💾 Arquivo 'mushroom_processed_conservative.csv' salvo com sucesso!")

    # 8. VISUALIZAÇÃO DA DISTRIBUIÇÃO FINAL
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    class_distribution.plot(kind='bar', color=['lightgreen', 'lightcoral'])
    plt.title('Distribuição das Classes\n(Dataset Limpo)')
    plt.xlabel('Classe')
    plt.ylabel('Quantidade')
    plt.xticks(ticks=[0, 1], labels=['Comestível (e)', 'Venenoso (p)'], rotation=0)

    plt.subplot(1, 3, 2)
    missing_before = missing_info[missing_info['Valores Faltantes'] > 0]['Valores Faltantes'].sum()
    missing_after = 0
    plt.bar(['Antes', 'Depois'], [missing_before, missing_after], color=['orange', 'green'])
    plt.title('Dados Faltantes\nAntes vs Depois do Pré-processamento')
    plt.ylabel('Quantidade de Valores Faltantes')

    plt.subplot(1, 3, 3)
    sizes = [len(df_clean), len(df) - len(df_clean)]
    labels = ['Mantidas', 'Removidas']
    colors = ['lightblue', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Proporção de Instâncias\nMantidas vs Removidas')

    plt.tight_layout()
    plt.savefig('preprocessing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return df_final, X_encoded, y_encoded, df_clean

# EXECUTAR O PRÉ-PROCESSAMENTO
if __name__ == "__main__":
    df_final, X_encoded, y_encoded, df_clean = conservative_preprocessing()

    # RESUMO EXECUTIVO
    print("\n" + "=" * 70)
    print("📋 RESUMO EXECUTIVO DO PRÉ-PROCESSAMENTO")
    print("=" * 70)
    print("✓ Abordagem: Conservadora (remoção de instâncias incompletas)")
    print("✓ Justificativa: Preservar confiabilidade em problema de saúde")
    print("✓ Dataset final garantido: Sem valores faltantes")
    print("✓ Balanceamento mantido: Verificado")
    print("✓ Método: Transparente e reprodutível")
    print("=" * 70)
