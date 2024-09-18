import pandas as pd
import numpy as np

# Membaca file CSV
df = pd.read_csv("PASIEN.csv")

# pembentukan dataset
df_new = {}
df_old = df.to_dict()
df_new['X1'] = df_old['UMUR']
df_new['X2'] = df_old['BD']
df_new['X3'] = df_old['GDS']
df_new['X4'] = df_old['TINGKAT PENYAKIT']
df_new['X5'] = df_old['X1']
df_new['X6'] = df_old['X2']
df_new['X7'] = df_old['X3']
df_new['X8'] = df_old['X4']
df_new['X9'] = df_old['X5']
df_new['X10'] = df_old['X9']
df_new['X11'] = df_old['X6']
df_new['X12'] = df_old['X7']
df_new['X13'] = df_old['X8']
df_new['X14'] = df_old['X12']
df_new['X15'] = df_old['X15']
df_new['X16'] = df_old['X17']
df_new['X17'] = df_old['X18']
df_new['y'] = df_old['STATUS']
df = pd.DataFrame(df_new)

# X1 (umur) tahun
x1 = df['X1'].values
x1_new = pd.Series([0 if x < 45 else 1 for x in x1])
df['X1'] = x1_new.astype(float)

# X2 (berat badan) Kilogram
x2 = df['X2'].values
x2_new = pd.Series([0 if x < 59 else 1 for x in x2])
df['X2'] = x2_new.astype(float)

# X3 (gula darah sewaktu) Miligram/Desiliter
x3 = df['X3'].values
x3_new = pd.Series([0 if x <= 60 else 0.5 if x > 60 and x < 200 else 1 for x in x3])
df['X3'] = x3_new

# X4 (penyakit penyerta)
x4 = df['X4'].values
x4_new = pd.Series([0 if x == 1 else 0.25 if x == 2 else 0.75 if x == 3 else 1 for x in x4])
df['X4'] = x4_new

# X5 (tingkat kesadaran pembukaan mata) 
x5 = df['X5'].values
x5_new = pd.Series([0 if x == 4 else 0.25 if x == 3 else 0.75 if x == 2 else 1 for x in x5])
df['X5'] = x5_new

# X6 (tingkat kesadaran motorik) 
x6 = df['X6'].values
x6_new = pd.Series([0 if x == 6 else 0.20 if x == 5 else 0.40 if x == 4 else 0.60 if x == 3 else 0.80 if x == 2 else 1 for x in x6])
df['X6'] = x6_new

# X7 (tingkat kesadaran lisan) 
x7 = df['X7'].values
x7_new = pd.Series([0 if x == 5 else 0.25 if x == 4 else 0.50 if x == 3 else 0.75 if x == 2 else 1 for x in x7])
df['X7'] = x7_new

# X8 (denyut nadi) 
x8 = df['X8'].values
x8_new = pd.Series([0 if x < 100 else 1 for x in x8])
df['X8'] = x8_new.astype(float)

# X9 (tekanan darah sitolik) 
x9 = df['X9'].values
x9_new = pd.Series([0 if x < 120 else 0.50 if x >= 120 and x < 140 else 1 for x in x9])
df['X9'] = x9_new

# X10 (demam penetap) 
x10 = df['X10'].values
x10_new = pd.Series([0 if x == 'ada' else 1 for x in x10])
df['X10'] = x10_new.astype(float)

# X11 (kehilangan pendengaran/penglihatan mendadak) 
x11 = df['X11'].values
x11_new = pd.Series([0 if x == 'ada' else 1 for x in x11])
df['X11'] = x11_new.astype(float)

# X12 (kelumpuhan mendadak) 
x12 = df['X12'].values
x12_new = pd.Series([0 if x == 'ada' else 1 for x in x12])
df['X12'] = x12_new.astype(float)

# X13 (pendarahan aktif) 
x13 = df['X13'].values
x13_new = pd.Series([0 if x == 'ada' else 1 for x in x13])
df['X13'] = x13_new.astype(float)

# X14 (eviserasi atau dishisensi luka)
x14 = df['X14'].values
x14_new = pd.Series([0 if x == 'ada' else 1 for x in x14])
df['X14'] = x14_new.astype(float)

# X15 (monitoring tanda vital) 
x15 = df['X15'].values
x15_new = pd.Series([0 if x == 'ya' else 1 for x in x15])
df['X15'] = x15_new.astype(float)

# X16 (anbiotika intramuskuler/intravena setiap 8 jam)
x16 = df['X16'].values
x16_new = pd.Series([0 if x == 'ada' else 1 for x in x16])
df['X16'] = x16_new.astype(float)

# X17 (pemakaian resporator kontinyu) 
x17 = df['X17'].values
x17_new = pd.Series([0 if x == 'ada' else 1 for x in x17])
df['X17'] = x17_new.astype(float)

# y (jalan atau inap)
y = df['y'].values
y_new = pd.Series([0 if char.startswith("JALAN") else 1 for char in y])
df['y'] = y_new

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    X (pd.DataFrame or np.array): Features
    y (pd.Series or np.array): Target variable
    test_size (float): Proportion of the dataset to include in the test split (0 to 1)
    random_state (int): Random seed for reproducibility
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Ensure X and y have the same number of samples
    assert len(X) == len(y), "X and y must have the same number of samples"
    
    # Get the number of samples
    n_samples = len(X)
    
    # Calculate the number of test samples
    n_test = int(n_samples * test_size)
    
    # Create an array of indices and shuffle it
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split the indices into train and test
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split X and y
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]
    
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

class InputLayer:
    def __init__(self, input_data) -> None:
        self.output = input_data

class PatternLayer:
    """
    Pattern Layer menghitung kesamaan (similarity) 
    antara data input dan setiap vektor pelatihan menggunakan fungsi Gaussian kernel. 
    Setiap neuron di Pattern Layer merepresentasikan satu vektor pelatihan.
    """
    def __init__(self, X_train, sigma):
        self.X_train = X_train  # Vektor pelatihan
        self.sigma = sigma      # Parameter bandwith untuk gaussian kernel

    def gaussian_kernel(self, x, xi):
        return np.exp(-np.linalg.norm(x - xi)**2 / (2 * (self.sigma ** 2)))
    
    def output(self, x):
        # Menghasilkan output berupa array dari Gaussian kernel untuk setiap neuron        
        return np.array([self.gaussian_kernel(x, xi) for xi in self.X_train])
    

class SummationLayer:
    """
    Summation Layer mengelompokkan output dari Pattern Layer 
    berdasarkan kelas dan menjumlahkan probabilitas untuk setiap kelas.
    """

    def __init__(self, y_train):
        self.y_train = y_train

    def output(self, pattern_outputs):
        classes = np.unique(self.y_train)
        class_nums = np.zeros(len(classes))

        for i, c in enumerate(classes):
            # menjumlahkan probabilitas untuk kelas tertentu
            class_nums[i] = np.sum(pattern_outputs[self.y_train == c])

        return class_nums / np.sum(class_nums) # Normalisasi output
    
    
class OutputLayer:
    """
    Output Layer memilih kelas dengan probabilitas tertinggi sebagai prediksi akhir.
    """

    def __init__(self, classes):
        self.classes = classes

    def output(self, summation_output):
        return self.classes[np.argmax(summation_output)]
    
class PNN:
    """
    Menggabungkan semua layer untuk membentuk model PNN
    """

    def __init__(self, X_train, y_train, sigma):
        self.input_layer = InputLayer(None)
        self.pattern_layer = PatternLayer(X_train, sigma)
        self.summation_layer = SummationLayer(y_train)
        self.output_layer = OutputLayer(np.unique(y_train))

    def predict(self, x_test):
        self.input_layer.output = x_test
        pattern_output = self.pattern_layer.output(self.input_layer.output)
        summation_output = self.summation_layer.output(pattern_output)
        return self.output_layer.output(summation_output)
    
    def predict_dataset(self, X_test):
        return np.array([self.predict(x) for x in X_test])


def calculate_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(y_true)):
        true_index = class_to_index[y_true[i]]
        pred_index = class_to_index[y_pred[i]]
        cm[true_index][pred_index] += 1
    
    return cm, classes

def calculate_accuracy(cm):
    return np.sum(np.diag(cm)) / np.sum(cm)

def calculate_class_accuracy(cm):
    # Akurasi untuk setiap kelas = True Positives / Total Instances of the Class
    class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    class_accuracy = np.where(np.sum(cm, axis=1) == 0, 0, class_accuracy)
    return class_accuracy

# def calculate_precision(cm):
#     return np.diag(cm) / np.sum(cm, axis=0)

# def calculate_recall(cm):
#     return np.diag(cm) / np.sum(cm, axis=1)

def calculate_precision(cm):
    precision = np.diag(cm) / np.sum(cm, axis=0)
    precision = np.where(np.sum(cm, axis=0) == 0, 0, precision)
    return precision

def calculate_recall(cm):
    recall = np.diag(cm) / np.sum(cm, axis=1)
    recall = np.where(np.sum(cm, axis=1) == 0, 0, recall)
    return recall

def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-10)

# contoh penggunaan model
from sklearn.metrics import accuracy_score

X = df.drop(columns='y').values
y = df['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sigma = 1.0

def cross_validation_PNN(model: PNN, X, y, k=5):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    all_results = []
    for i in range(k):
        test_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        
        X_train, y_train = X[train_indices], y[train_indices].astype(float)
        X_test, y_test = X[test_indices], y[test_indices].astype(float)
        
        X_train = X_train.astype(float)
        y_train = y_train.astype(float)
        PNN = model(X_train, y_train, sigma)
        
        # if isinstance(model, SimpleLGBM):
        #     predictions = (model.predict(X_test) > 0.5).astype(int)
        # else:
        
        predictions = PNN.predict_dataset(X_test)
        
        cm, classes = calculate_confusion_matrix(y_test, predictions)
        accuracy = calculate_accuracy(cm)
        precision = calculate_precision(cm)
        recall = calculate_recall(cm)
        f1_score = calculate_f1_score(precision, recall)
        
        all_results.append({
            'fold': i+1,
            'cm': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
    
    return all_results

res = cross_validation_PNN(PNN, X, y)
print(res)

# model = PNN(X_train, y_train, sigma)

# y_pred = model.predict_dataset(X_test)

# y_train_series = pd.Series(y_test)
# print(y_train_series.value_counts())

# cm, classes = calculate_confusion_matrix(y_test, y_pred)
# print(f"Confusion Matrix\n {cm}")

# overall_acc = calculate_accuracy(cm)
# print(f"Overall Accuracy: {overall_acc * 100:.2f}")

# each_class_acc = calculate_class_accuracy(cm)
# print(f"Accuracy each Class: {each_class_acc}")

# pre = calculate_precision(cm)
# print(f"Precision each Class: {pre}")

# rec = calculate_recall(cm)
# print(f"Recall each Class: {rec}")

# f1 = calculate_f1_score(pre, rec)
# print(f"F1-Score each Class: {f1}")

