import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Ignorar avisos para manter o log limpo
warnings.filterwarnings('ignore')

# 1. CARREGAR DADOS PROCESSADOS
print("Carregando dataset...")
df = pd.read_csv('mushroom_processed_conservative.csv')

# CORREÇÃO: Remover explicitamente quaisquer linhas com NaN na coluna 'target' após o carregamento
# Isso garante que X e y não contenham NaNs, caso o arquivo CSV esteja corrompido ou tenha sido salvo incorretamente
df_cleaned_for_training = df.dropna(subset=['target'])
X = df_cleaned_for_training.drop('target', axis=1)
y = df_cleaned_for_training['target'].astype(int) # Garante que y seja int para StratifiedKFold

print(f"Dataset shape: {X.shape}")
print(f"Distribuição das classes: {y.value_counts().to_dict()}")

# 2. CONFIGURAÇÃO DA VALIDAÇÃO CRUZADA
cv_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 3. DEFINIÇÃO DOS MODELOS E HIPERPARÂMETROS
models_config = [
    {
        'name': 'Árvore de Decisão',
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 5],
            'min_samples_split': [2, 5]
        }
    },
    {
        'name': 'KNN (Vizinhos Próximos)',
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    },
    {
        'name': 'Naive Bayes (Gaussiano)',
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    },
    {
        'name': 'Regressão Logística',
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    {
        'name': 'Rede Neural MLP',
        'model': MLPClassifier(max_iter=500, random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001]
        }
    }
]

# 4. LOOP DE TREINAMENTO E AVALIAÇÃO SEM VAZAMENTO
results_data = []

print(f"{'='*80}")
print(f"{'INICIANDO O EXPERIMENTO - SEM VAZAMENTO':^80}")
print(f"{'='*80}")

for config in models_config:
    model_name = config['name']
    model = config['model']
    params = config['params']

    print(f"\n> Processando: {model_name}")
    print(f"  Combinções de parâmetros: {sum(len(v) if isinstance(v, list) else 1 for v in params.values())}")

    # Arrays para armazenar resultados de cada fold
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    best_params_per_fold = []

    # GridSearch dentro de cada fold para evitar vazamento
    for fold, (train_idx, test_idx) in enumerate(cv_stratified.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # A. OTIMIZAÇÃO DE PARÂMETROS (GridSearch apenas nos dados de treino do fold)
        grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params_per_fold.append(grid.best_params_)

        # B. AVALIAÇÃO NO CONJUNTO DE TESTE
        y_pred = best_model.predict(X_test)

        # Calcular métricas
        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_precisions.append(precision_score(y_test, y_pred, zero_division=0))
        fold_recalls.append(recall_score(y_test, y_pred, zero_division=0))
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))

    # Encontrar os parâmetros mais frequentes entre os folds
    from collections import Counter
    param_counts = Counter(str(params) for params in best_params_per_fold)
    most_common_params = eval(param_counts.most_common(1)[0][0])

    # Cálculo das Médias e Desvios Padrão
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    mean_prec = np.mean(fold_precisions)
    std_prec = np.std(fold_precisions)

    mean_rec = np.mean(fold_recalls)
    std_rec = np.std(fold_recalls)

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    print(f"  Resultados (Média ± Desvio Padrão) - 10 folds:")
    print(f"  - Acurácia:  {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  - Precision: {mean_prec:.4f} ± {std_prec:.4f}")
    print(f"  - Recall:    {mean_rec:.4f} ± {std_rec:.4f}")
    print(f"  - F1-Score:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  - Parâmetros mais frequentes: {most_common_params}")

    # Armazenar os resultados
    results_data.append({
        'Algoritmo': model_name,
        'Acurácia': f"{mean_acc:.4f} ± {std_acc:.4f}",
        'Precision': f"{mean_prec:.4f} ± {std_prec:.4f}",
        'Recall': f"{mean_rec:.4f} ± {std_rec:.4f}",
        'F1-Score': f"{mean_f1:.4f} ± {std_f1:.4f}",
        'Melhores Params': str(most_common_params)
    })

# 5. RELATÓRIO FINAL
print(f"\n{'='*80}")
print(f"{'TABELA RESUMO - VALIDAÇÃO CRUZADA 10-FOLD':^80}")
print(f"{'='*80}")

results_df = pd.DataFrame(results_data)

# Exibir tabela formatada
print("\n" + results_df.drop('Melhores Params', axis=1).to_markdown(index=False))

# Exibir parâmetros separadamente
print(f"\n{'='*80}")
print(f"{'MELHORES PARÂMETROS POR ALGORITMO':^80}")
print(f"{'='*80}")
for result in results_data:
    print(f"\n{result['Algoritmo']}:")
    print(f"  {result['Melhores Params']}")

# Resultados completos em CSV
results_df.to_csv('resultados_finais_sem_vazamento.csv', index=False)
print(f"\n{'='*80}")
print("Arquivo 'resultados_finais_sem_vazamento.csv' salvo com sucesso!")

# 6. ANÁLISE COMPARATIVA
print(f"\n{'='*80}")
print(f"{'ANÁLISE COMPARATIVA':^80}")
print(f"{'='*80}")

# Encontrar melhor algoritmo por métrica
metrics = ['Acurácia', 'Precision', 'Recall', 'F1-Score']
for metric in metrics:
    best_idx = None
    best_value = 0
    for i, result in enumerate(results_data):
        value = float(result[metric].split(' ± ')[0])
        if value > best_value:
            best_value = value
            best_idx = i

    if best_idx is not None:
        print(f"Melhor {metric}: {results_data[best_idx]['Algoritmo']} ({best_value:.4f})")

# Verificar se há overfitting (alta acurácia com alta variância)
print(f"\n{'ANÁLISE DE OVERFITTING':^80}")
print(f"{'='*80}")
for result in results_data:
    acc_value = float(result['Acurácia'].split(' ± ')[0])
    acc_std = float(result['Acurácia'].split(' ± ')[1])

    if acc_std > 0.05:  # Desvio padrão maior que 5%
        status = "⚠️  POSSÍVEL OVERFITTING"
    elif acc_std > 0.02:
        status = "ℹ️  VARIÂNCIA MODERADA"
    else:
        status = "✅ ESTÁVEL"

    print(f"{result['Algoritmo']:.<30} {acc_value:.4f} ± {acc_std:.4f} {status}")
