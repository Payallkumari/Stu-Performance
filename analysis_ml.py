# analysis_ml.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

DATA_PATH = os.path.join("data", "StudentsPerformance.csv")

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def task2_avg_by_gender(df):
    avg_scores = df.groupby('gender')[['math score','reading score','writing score']].mean()
    print("Average scores by gender:\n", avg_scores)
    # plot bar chart
    ax = avg_scores.plot(kind='bar', rot=0, title='Average Scores by Gender')
    ax.set_ylabel('Average Score')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'avg_scores_by_gender.png'))
    plt.close()
    return avg_scores

def task3_classification(df):
    # 3a: define Result
    df = df.copy()
    df['average'] = df[['math score','reading score','writing score']].mean(axis=1)
    df['Result'] = (df['average'] >= 50).astype(int)  # 1=Pass, 0=Fail

    X = df[['math score','reading score','writing score']]
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3c train SVM
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", acc)

    return model, cm, acc

def task4_kmeans(df, n_clusters=3):
    X = df[['math score','reading score','writing score']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    df2 = df.copy()
    df2['cluster'] = labels

    # scatter math vs reading colored by cluster
    plt.figure(figsize=(8,6))
    plt.scatter(df2['math score'], df2['reading score'], c=labels)
    plt.xlabel('Math Score')
    plt.ylabel('Reading Score')
    plt.title(f'KMeans Clusters (k={n_clusters})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'kmeans_k{n_clusters}.png'))
    plt.close()

    return kmeans, df2

def elbow_method(df, k_max=10):
    X = df[['math score','reading score','writing score']].values
    inertias = []
    ks = list(range(1, k_max+1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure()
    plt.plot(ks, inertias, 'o-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'elbow_method.png'))
    plt.close()
    return ks, inertias

def task5_regression(df):
    X = df[['reading score']].values
    y = df['writing score'].values
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)

    # plot
    plt.figure(figsize=(8,6))
    plt.scatter(X, y)
    # regression line sorted for clear line
    order = np.argsort(X[:,0])
    plt.plot(X[order], y_pred[order])
    plt.xlabel('Reading Score')
    plt.ylabel('Writing Score')
    plt.title('Linear Regression - Predict Writing from Reading')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'regression_reading_writing.png'))
    plt.close()

    print("Linear Regression coefficient:", reg.coef_[0])
    print("Intercept:", reg.intercept_)
    print("R^2 score:", r2)
    return reg, r2

if __name__ == "__main__":
    df = load_data()
    task2_avg_by_gender(df)
    model, cm, acc = task3_classification(df)
    kmeans, df_clusters = task4_kmeans(df, n_clusters=3)
    ks, inertias = elbow_method(df, k_max=8)
    reg, r2 = task5_regression(df)
