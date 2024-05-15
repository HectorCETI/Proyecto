import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Función para cargar o generar los conjuntos de datos
def load_or_generate_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Generar el conjunto de datos si no está disponible
        # Aquí puedes colocar la lógica para generar los datos
        pass
    return df

# Función para entrenar y evaluar un modelo
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Inicializar los clasificadores
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machines": SVC(),
        "Naive Bayes": GaussianNB()
    }
    
    # Resultados de cada método en el conjunto de datos
    results = {}
    
    # Iterar sobre cada clasificador
    for name, clf in classifiers.items():
        # Entrenar el clasificador
        clf.fit(X_train, y_train)
        # Realizar predicciones
        y_pred = clf.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        confusion = confusion_matrix(y_test, y_pred)
        
        # Almacenar resultados
        results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion': confusion}
    
    # Imprimir resultados y comparativa
    print("\nResults Comparison:")
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1 Score: {metrics['f1']}")
        print(f"Confusion Matrix:\n{metrics['confusion']}")
    
    # Comparativa para determinar el mejor método
    best_method = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest Method for this dataset: {best_method}")

# Generar o cargar los conjuntos de datos
datasets = ["zoo.csv", "zoo2.csv", "zoo3.csv"]
for i, file_path in enumerate(datasets):
    print(f"\nDataset {i+1}: {file_path}")
    df = load_or_generate_dataset(file_path)
    
    # Dividir el conjunto de datos en características (X) y etiquetas (y)
    X = df.drop(['animal_name', 'class_type'], axis=1)
    y = df['class_type']
    
    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar y evaluar el modelo
    train_and_evaluate_model(X_train, X_test, y_train, y_test)
