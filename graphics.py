import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.gridspec as gridspec

# Configurações de estilo
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def create_comprehensive_plots(results_data, X, y, models_config, cv_stratified):
    """
    Cria todos os gráficos essenciais para o artigo
    """

    # 1. GRÁFICO DE COMPARAÇÃO DE ALGORITMOS (BARRAS)
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # 1.1 Comparação de Métricas Principais
    ax1 = fig.add_subplot(gs[0, :])
    metrics_data = []
    for result in results_data:
        metrics_data.append({
            'Algoritmo': result['Algoritmo'],
            'Acurácia': float(result['Acurácia'].split(' ± ')[0]),
            'Precision': float(result['Precision'].split(' ± ')[0]),
            'Recall': float(result['Recall'].split(' ± ')[0]),
            'F1-Score': float(result['F1-Score'].split(' ± ')[0])
        })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Algoritmo', inplace=True)

    x = np.arange(len(metrics_df))
    width = 0.2

    bars1 = ax1.bar(x - width*1.5, metrics_df['Acurácia'], width, label='Acurácia', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x - width/2, metrics_df['Precision'], width, label='Precision', alpha=0.8, color='lightcoral')
    bars3 = ax1.bar(x + width/2, metrics_df['Recall'], width, label='Recall', alpha=0.8, color='lightgreen')
    bars4 = ax1.bar(x + width*1.5, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8, color='gold')

    ax1.set_xlabel('Algoritmos', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('COMPARAÇÃO DAS MÉTRICAS POR ALGORITMO', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.8, 1.02)

    # Adicionar valores nas barras
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. HEATMAP DE DESEMPENHO
    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.4f',
                linewidths=1, cbar_kws={'label': 'Score'}, ax=ax2)
    ax2.set_title('HEATMAP DE DESEMPENHO - MÉTRICAS POR ALGORITMO',
                  fontsize=12, fontweight='bold')

    # 3. GRÁFICO DE COMPARAÇÃO DE ACURÁCIA COM BARRAS DE ERRO
    ax3 = fig.add_subplot(gs[1, 1])
    algorithms = [result['Algoritmo'] for result in results_data]
    accuracies = [float(result['Acurácia'].split(' ± ')[0]) for result in results_data]
    errors = [float(result['Acurácia'].split(' ± ')[1]) for result in results_data]

    bars = ax3.bar(algorithms, accuracies, yerr=errors, capsize=5,
                   color='lightblue', alpha=0.7, edgecolor='navy', linewidth=1.5)

    ax3.set_ylabel('Acurácia', fontweight='bold')
    ax3.set_title('ACURÁCIA COM INTERVALO DE CONFIANÇA (DESVIO PADRÃO)',
                  fontsize=12, fontweight='bold')
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.85, 1.02)

    # Destacar a melhor acurácia
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkorange')

    # 4. MATRIZ DE CONFUSÃO DO MELHOR ALGORITMO
    ax4 = fig.add_subplot(gs[2, 0])

    # Encontrar e treinar o melhor algoritmo
    best_algo_name = algorithms[best_idx]
    best_config = next(config for config in models_config if config['name'] == best_algo_name)

    # Usar último fold para matriz de confusão
    for train_idx, test_idx in cv_stratified.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid = GridSearchCV(best_config['model'], best_config['params'], cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Comestível', 'Venenoso'],
                yticklabels=['Comestível', 'Venenoso'])
    ax4.set_title(f'MATRIZ DE CONFUSÃO - {best_algo_name}\n', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Predito')
    ax4.set_ylabel('Real')

    # 5. CURVAS ROC COMPARATIVAS
    ax5 = fig.add_subplot(gs[2, 1])

    for config in models_config:
        if config['name'] in ['Árvore de Decisão', 'Regressão Logística', 'Naive Bayes', 'MLP']:
            model = config['model']
            grid = GridSearchCV(model, config['params'], cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)

                ax5.plot(fpr, tpr, linewidth=2,
                        label=f'{config["name"]} (AUC = {roc_auc:.3f})')

    ax5.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Classificador Aleatório')
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, 1.05])
    ax5.set_xlabel('Taxa de Falsos Positivos', fontweight='bold')
    ax5.set_ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
    ax5.set_title('CURVAS ROC - COMPARAÇÃO ENTRE ALGORITMOS', fontsize=12, fontweight='bold')
    ax5.legend(loc="lower right")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparacao_completa_algoritmos.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. GRÁFICO DE DISTRIBUIÇÃO DAS ACURÁCIAS (BOXPLOT)
    plt.figure(figsize=(14, 8))

    # Coletar acurácias de todos os folds para cada algoritmo
    accuracy_data = []

    for config in models_config:
        model_name = config['name']
        model = config['model']
        params = config['params']

        fold_accuracies = []
        for train_idx, test_idx in cv_stratified.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            grid = GridSearchCV(model, params, cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            fold_accuracies.append(accuracy_score(y_test, y_pred))

        for acc in fold_accuracies:
            accuracy_data.append({'Algoritmo': model_name, 'Acurácia': acc})

    accuracy_df = pd.DataFrame(accuracy_data)

    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=accuracy_df, x='Algoritmo', y='Acurácia', palette='Set2')
    plt.title('DISTRIBUIÇÃO DAS ACURÁCIAS POR ALGORITMO\n(BOXPLOT)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(data=accuracy_df, x='Algoritmo', y='Acurácia', palette='Set2')
    plt.title('DISTRIBUIÇÃO DAS ACURÁCIAS POR ALGORITMO\n(VIOLIN PLOT)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('distribuica_acuracia.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 7. GRÁFICO DE RADAR (SPIDER CHART) PARA COMPARAÇÃO MULTIDIMENSIONAL
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Métricas para o radar chart
    metrics_radar = ['Acurácia', 'Precision', 'Recall', 'F1-Score', 'Estabilidade']
    N = len(metrics_radar)

    # Ângulos para cada métrica
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fechar o círculo

    # Plot cada algoritmo
    for i, result in enumerate(results_data):
        values = [
            float(result['Acurácia'].split(' ± ')[0]),
            float(result['Precision'].split(' ± ')[0]),
            float(result['Recall'].split(' ± ')[0]),
            float(result['F1-Score'].split(' ± ')[0]),
            1 - float(result['Acurácia'].split(' ± ')[1])  # Estabilidade (1 - desvio padrão)
        ]
        values += values[:1]  # Fechar o círculo

        ax.plot(angles, values, linewidth=2, label=result['Algoritmo'])
        ax.fill(angles, values, alpha=0.1)

    # Configurar o gráfico radar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar)
    ax.set_ylim(0, 1)
    ax.set_title('COMPARAÇÃO MULTIDIMENSIONAL - RADAR CHART', size=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig('radar_chart_comparacao.png', dpi=300, bbox_inches='tight')
    plt.show()

    return metrics_df

# USO DA FUNÇÃO (adicionar ao final do seu código principal)
print(f"\n{'='*80}")
print(f"{'GERANDO GRÁFICOS PARA ANÁLISE':^80}")
print(f"{'='*80}")

# Chamar a função de geração de gráficos
metrics_summary = create_comprehensive_plots(results_data, X, y, models_config, cv_stratified)

print("✅ Todos os gráficos gerados e salvos com sucesso!")
