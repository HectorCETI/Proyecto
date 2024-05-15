import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
def train_and_evaluate_model(X_train, X_test, y_train, y_test, classifier_name):
    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machines": SVC(),
        "Naive Bayes": GaussianNB()
    }
    
    clf = classifiers[classifier_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
        
    accuracy, precision, recall, f1, confusion = evaluate_model(y_test, y_pred)
        
    print(f"\n{classifier_name} Results:")
    print_metrics(accuracy, precision, recall, f1, confusion)
    
    plot_confusion_matrix(confusion, classifier_name)

# Función para evaluar un modelo y obtener métricas
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion = confusion_matrix(y_true, y_pred)
    
    return accuracy, precision, recall, f1, confusion

# Función para imprimir métricas
def print_metrics(accuracy, precision, recall, f1, confusion):
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{confusion}")

# Función para graficar la matriz de confusión
def plot_confusion_matrix(confusion, classifier_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {classifier_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Generar o cargar los conjuntos de datos
datasets = ["zoo.csv", "zoo2.csv", "zoo3.csv"]
for i, file_path in enumerate(datasets):
    print(f"\nDataset {i+1}: {file_path}")
    df = load_or_generate_dataset(file_path)
    X = df.drop(['animal_name', 'class_type'], axis=1)
    y = df['class_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for classifier_name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machines", "Naive Bayes"]:
        train_and_evaluate_model(X_train, X_test, y_train, y_test, classifier_name)
