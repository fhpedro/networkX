# -*- coding: utf-8 -*-
"""Lista de exercicio 3.ipynb"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Definição de cores
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
CIANO = "\033[96m"
YELLOW = "\033[93m"

def processar_dataset(caminho_csv):
    """Versão que processa as colunas categóricas"""
    print(f"\nProcessando o arquivo: {YELLOW}{caminho_csv}{RESET}")
    
    # Ler o CSV
    df = pd.read_csv(caminho_csv)
    
    print(f"  Dimensões originais: {YELLOW}{df.shape}{RESET}")
    print(f"  Colunas originais: {CIANO}{list(df.columns)}{RESET}")
    
    # 1. REMOVER COLUNA 'Unnamed: 0' (índice sequencial)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print(f"  {GREEN}✓{RESET} Removida coluna 'Unnamed: 0' (índice)")
    
    # 2. REMOVER VALORES FALTANTES
    df = df.dropna()
    print(f"  Número de observações após remover valores faltantes: {YELLOW}{len(df)}{RESET}")
    
    
    # Separar target (AHD) das features categóricas
    target_col = 'AHD'
    
    # Fazer encoding das variáveis categóricas
    label_encoders = {}
    
    # Encoding do target
    if target_col in df.columns:
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col])
        label_encoders[target_col] = le_target
        print(f"  {GREEN}✓{RESET} Encoded target '{target_col}': {le_target.classes_}")
    
    print(f"  Dimensões finais: {YELLOW}{df.shape}{RESET}")
    print(f"  Colunas finais: {CIANO}{list(df.columns)}{RESET}")
    
    # Dropar colunas 'ChestPain' e 'Thal', se existirem
    colunas_para_dropar = ['ChestPain', 'Thal']
    df = df.drop(columns=[col for col in colunas_para_dropar if col in df.columns], errors='ignore')
    print(f"  {GREEN}✓{RESET} Dropped colunas: {CIANO}{colunas_para_dropar}{RESET}")


    # Dividir em features (X) e target (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"  Features: {YELLOW}{X.shape}{RESET}")
    print(f"  Target: {YELLOW}{y.shape}{RESET}")
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Resultados
    resultados = {}
    
    print(f"\n  {BLUE}Treinando modelos...{RESET}")
    
    # Naive Bayes
    try:
        clf_nb = GaussianNB().fit(X_train, y_train)
        resultados['Naive Bayes'] = {
            'AcertoIn': clf_nb.score(X_train, y_train),
            'AcertoOut': clf_nb.score(X_test, y_test)
        }
        print(f"    {GREEN}✓{RESET} Naive Bayes")
    except Exception as e:
        print(f"    {RED}✗{RESET} Naive Bayes: {str(e)}")
        resultados['Naive Bayes'] = {'Erro': str(e)}
    
    # Regressão Logística
    try:
        clf_lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
        resultados['Logistic Regression'] = {
            'AcertoIn': clf_lr.score(X_train, y_train),
            'AcertoOut': clf_lr.score(X_test, y_test)
        }
        print(f"    {GREEN}✓{RESET} Logistic Regression")
    except Exception as e:
        print(f"    {RED}✗{RESET} Logistic Regression: {str(e)}")
        resultados['Logistic Regression'] = {'Erro': str(e)}
    
    # Árvore de Decisão
    try:
        clf_dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        resultados['Decision Tree'] = {
            'AcertoIn': clf_dt.score(X_train, y_train),
            'AcertoOut': clf_dt.score(X_test, y_test)
        }
        print(f"    {GREEN}✓{RESET} Decision Tree")
    except Exception as e:
        print(f"    {RED}✗{RESET} Decision Tree: {str(e)}")
        resultados['Decision Tree'] = {'Erro': str(e)}
    
    # Redes Neurais - Modelo 1 (random_state=1)
    try:
        clf_nn1 = MLPClassifier(random_state=1, max_iter=1000).fit(X_train, y_train)
        resultados['Neural Network (random_state=1)'] = {
            'AcertoIn': clf_nn1.score(X_train, y_train),
            'AcertoOut': clf_nn1.score(X_test, y_test)
        }
        print(f"    {GREEN}✓{RESET} Neural Network (rs=1)")
    except Exception as e:
        print(f"    {RED}✗{RESET} Neural Network (rs=1): {str(e)}")
        resultados['Neural Network (random_state=1)'] = {'Erro': str(e)}
    
    # Redes Neurais - Modelo 2 (random_state=22)
    try:
        clf_nn2 = MLPClassifier(random_state=22, max_iter=1000).fit(X_train, y_train)
        resultados['Neural Network (random_state=22)'] = {
            'AcertoIn': clf_nn2.score(X_train, y_train),
            'AcertoOut': clf_nn2.score(X_test, y_test)
        }
        print(f"    {GREEN}✓{RESET} Neural Network (rs=22)")
    except Exception as e:
        print(f"    {RED}✗{RESET} Neural Network (rs=22): {str(e)}")
        resultados['Neural Network (random_state=22)'] = {'Erro': str(e)}
    
    # Redes Neurais - Modelo 3 (max_iter=2000, random_state=11)
    try:
        clf_nn3 = MLPClassifier(max_iter=2000, random_state=11).fit(X_train, y_train)
        resultados['Neural Network (max_iter=2000, random_state=11)'] = {
            'AcertoIn': clf_nn3.score(X_train, y_train),
            'AcertoOut': clf_nn3.score(X_test, y_test)
        }
        print(f"    {GREEN}✓{RESET} Neural Network (max_iter=2000)")
    except Exception as e:
        print(f"    {RED}✗{RESET} Neural Network (max_iter=2000): {str(e)}")
        resultados['Neural Network (max_iter=2000, random_state=11)'] = {'Erro': str(e)}
    
    # Calcular erros para modelos que funcionaram
    for modelo, valores in resultados.items():
        if 'Erro' not in valores:
            valores['ErroIn'] = 1 - valores['AcertoIn']
            valores['ErroOut'] = 1 - valores['AcertoOut']
    
    return resultados, label_encoders

def main():
    print(f"{CIANO}{'='*80}{RESET}")
    print(f"{YELLOW}LISTA DE EXERCÍCIO 3{RESET}")
    print(f"{CIANO}{'='*80}{RESET}")
    
    # Caminhos dos datasets
    datasets = ['dataset1.csv', 'dataset2.csv', 'dataset3.csv']
    
    # Processar cada dataset e armazenar os resultados
    todos_resultados = {}
    todos_encoders = {}
    
    for dataset in datasets:
        try:
            resultados, encoders = processar_dataset(dataset)
            todos_resultados[dataset] = resultados
            todos_encoders[dataset] = encoders
        except Exception as e:
            print(f"{RED}Erro ao processar {dataset}: {str(e)}{RESET}")
            continue
    
    # Comparar os resultados
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{YELLOW}COMPARAÇÃO DOS RESULTADOS{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    
    for dataset, resultados in todos_resultados.items():
        print(f"\n{YELLOW}Resultados para {dataset}:{RESET}")
        for modelo, valores in resultados.items():
            if 'Erro' in valores:
                print(f"  {modelo}: {RED}ERRO - {valores['Erro']}{RESET}")
            else:
                print(f"  {modelo}:")
                print(f"    Acerto dentro da amostra: {GREEN}{valores['AcertoIn']:.4f}{RESET}")
                print(f"    Acerto fora da amostra: {GREEN}{valores['AcertoOut']:.4f}{RESET}")
                print(f"    Erro dentro da amostra: {RED}{valores['ErroIn']:.4f}{RESET}")
                print(f"    Erro fora da amostra: {RED}{valores['ErroOut']:.4f}{RESET}")
    
    # Analisar os resultados para identificar overfitting, underfitting ou bom ajuste
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{YELLOW}ANÁLISE DE AJUSTE DOS MODELOS{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    
    for dataset, resultados in todos_resultados.items():
        print(f"\n{YELLOW}Análise para {dataset}:{RESET}")
        for modelo, valores in resultados.items():
            if 'Erro' in valores:
                print(f"  {modelo}: {RED}ERRO - {valores['Erro']}{RESET}")
                continue
                
            acerto_in = valores['AcertoIn']
            acerto_out = valores['AcertoOut']
            diferenca = acerto_in - acerto_out
            
            # Classificar o ajuste do modelo
            if diferenca > 0.1:  # Grande diferença entre treino e teste
                ajuste = f"{RED}Overfit (sobreajustado){RESET}"
            elif acerto_in < 0.7 and acerto_out < 0.7:  # Baixas acurácias em ambos
                ajuste = f"{RED}Underfit (subajustado){RESET}"
            else:  # Acurácias altas e próximas
                ajuste = f"{GREEN}Bem ajustado{RESET}"
            
            print(f"  {modelo}:")
            print(f"    Acerto dentro da amostra: {GREEN}{acerto_in:.4f}{RESET}")
            print(f"    Acerto fora da amostra: {GREEN}{acerto_out:.4f}{RESET}")
            print(f"    Diferença: {YELLOW}{diferenca:.4f}{RESET}")
            print(f"    Ajuste: {ajuste}")
    

if __name__ == "__main__":
    main()
