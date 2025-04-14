import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.font_manager as fm


# MNIST 데이터셋을 로드하는 함수
def load_mnist_data(train_size=20000, batch_size=500):
    """
    MNIST 데이터셋을 다운로드하고 로드
    
    Parameters:
    - train_size: 학습에 사용할 데이터의 크기
    - batch_size: 데이터 로더의 배치 크기
    
    Returns:
    - train_loader: 학습 데이터 로더
    - test_loader: 테스트 데이터 로더
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 학습 데이터 양 증가
    train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 커널 SVM 클래스 정의
class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=0.1, max_iter=100, tol=1e-3):
        """
        SVM 모델 초기화
        
        Parameters:
        - kernel: 사용할 커널 종류 (linear, rbf)
        - C: 규제 파라미터
        - gamma: RBF 커널의 감마 값
        - max_iter: 최대 반복 횟수
        - tol: 수렴 허용 오차
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0.0  # 절편

    def _kernel_function(self, x1, x2):
        """
        커널 함수 정의
        
        Parameters:
        - x1: 첫 번째 입력 데이터
        - x2: 두 번째 입력 데이터
        
        Returns:
        - 커널 행렬
        """
        if self.kernel == 'linear':
            return torch.matmul(x1, x2.T)
        elif self.kernel == 'rbf':
            x1_norm = torch.sum(x1**2, dim=1).view(-1, 1)
            x2_norm = torch.sum(x2**2, dim=1).view(1, -1)
            dist_squared = x1_norm + x2_norm - 2.0 * torch.matmul(x1, x2.T)
            return torch.exp(-self.gamma * dist_squared)
        else:
            raise ValueError(f"지원하지 않는 커널: {self.kernel}")

    def _smo(self, X, y):
        """
        SMO 알고리즘을 사용하여 SVM 모델을 학습
        
        Parameters:
        - X: 입력 데이터
        - y: 레이블
        
        Returns:
        - self: 학습된 SVM 모델
        """
        n_samples = X.shape[0]
        alpha = torch.zeros(n_samples, device=X.device)
        K = self._kernel_function(X, X)
        E = torch.zeros(n_samples, device=X.device)

        for iteration in range(self.max_iter):
            num_changed_alphas = 0
            
            for i in range(n_samples):
                f_i = torch.sum(alpha * y * K[:, i]) + self.b
                E[i] = f_i - y[i]

                if (y[i] * E[i] < -self.tol and alpha[i] < self.C) or (y[i] * E[i] > self.tol and alpha[i] > 0):
                    j = torch.argmin(E) if E[i] > 0 else torch.argmax(E)
                    if j == i:
                        continue

                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 0:
                        continue

                    alpha_j_old = alpha[j].clone()
                    alpha_i_old = alpha[i].clone()

                    L = max(0, alpha[i] + alpha[j] - self.C) if y[i] == y[j] else max(0, alpha[j] - alpha[i])
                    H = min(self.C, alpha[i] + alpha[j]) if y[i] == y[j] else min(self.C, self.C + alpha[j] - alpha[i])

                    if L == H:
                        continue

                    alpha[j] = alpha_j_old + y[j] * (E[i] - E[j]) / eta
                    alpha[j] = torch.clamp(alpha[j], L, H)
                    alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])

                    b1 = self.b - E[i] - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E[j] - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < alpha[i] < self.C:
                        self.b = b1
                    elif 0 < alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                print(f'SMO 알고리즘이 {iteration + 1}번째 반복에서 수렴했습니다.')
                break

        sv_indices = torch.where(alpha > 1e-5)[0]
        self.alpha = alpha[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]

        return self

    def fit(self, X, y):
        """
        SVM 모델 학습
        
        Parameters:
        - X: 입력 데이터
        - y: 레이블
        
        Returns:
        - self: 학습된 SVM 모델
        """
        self._smo(X, y)
        return self

    def predict(self, X):
        """
        입력 데이터에 대한 예측을 수행
        
        Parameters:
        - X: 입력 데이터
        
        Returns:
        - 예측된 레이블
        """
        if self.alpha is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit 메소드를 호출하세요.")

        K = self._kernel_function(X, self.support_vectors)
        decision = torch.matmul(K, self.alpha * self.support_vector_labels) + self.b
        return torch.sign(decision)

# 다중 클래스 분류를 위한 One-vs-Rest SVM 클래스 정의
class MulticlassSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=0.1, max_iter=100, tol=1e-3):
        """
        다중 클래스 SVM 모델 초기화
        
        Parameters:
        - kernel: 사용할 커널 종류 (linear, rbf)
        - C: 규제 파라미터
        - gamma: RBF 커널의 감마 값
        - max_iter: 최대 반복 횟수
        - tol: 수렴 허용 오차
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.models = {}

    def fit(self, X, y):
        """
        다중 클래스 SVM 모델 학습
        
        Parameters:
        - X: 입력 데이터
        - y: 레이블
        
        Returns:
        - self: 학습된 다중 클래스 SVM 모델
        """
        self.classes = torch.unique(y)

        for cls in self.classes:
            print(f"클래스 {cls}에 대한 SVM 학습 중...")
            binary_y = torch.where(y == cls, torch.tensor(1.0), torch.tensor(-1.0))
            model = KernelSVM(kernel=self.kernel, C=self.C, gamma=self.gamma, max_iter=self.max_iter, tol=self.tol)
            model.fit(X, binary_y)
            self.models[cls.item()] = model

        return self

    def predict(self, X):
        """
        입력 데이터에 대한 예측 수행
        
        Parameters:
        - X: 입력 데이터
        
        Returns:
        - 예측된 레이블
        """
        n_samples = X.shape[0]
        votes = torch.zeros((n_samples, len(self.classes)), device=X.device)

        for i, cls in enumerate(self.classes):
            model = self.models[cls.item()]
            K = model._kernel_function(X, model.support_vectors)
            decision = torch.matmul(K, model.alpha * model.support_vector_labels) + model.b
            votes[:, i] = decision

        predictions = torch.argmax(votes, dim=1)
        return predictions

# 학습 및 평가 실행 함수
def train_and_evaluate():
    """
    SVM 모델을 학습하고 평가
    
    Returns:
    - model: 학습된 SVM 모델
    - cm: 혼동 행렬
    """
    train_loader, test_loader = load_mnist_data()

    # 학습 데이터셋에서 배치를 추출하여 학습에 사용
    X_train = []
    y_train = []
    for batch_idx, (data, target) in enumerate(train_loader):
        X_train.append(data.reshape(data.shape[0], -1))
        y_train.append(target)
        if batch_idx >= 7:  # 8개의 배치 사용 (약 4000개)
            break

    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    # 테스트 데이터셋에서 배치를 추출하여 평가에 사용
    X_test = []
    y_test = []
    for batch_idx, (data, target) in enumerate(test_loader):
        X_test.append(data.reshape(data.shape[0], -1))
        y_test.append(target)
        if batch_idx >= 7:  # 8개의 배치 사용 (약 4000개)
            break

    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)

    # 데이터 정규화
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    # 다중 클래스 SVM 모델 생성 및 학습
    print("다중 클래스 SVM 모델 학습 중...")
    model = MulticlassSVM(kernel='rbf', C=1.0, gamma=0.01, max_iter=200, tol=0.01)
    model.fit(X_train, y_train)

    # 테스트 데이터로 예측
    print("테스트 데이터로 예측 중...")
    y_pred = model.predict(X_test)

    # 정확도 계산
    accuracy = torch.sum(y_pred == y_test).float() / y_test.size(0)
    print(f"테스트 정확도: {accuracy:.4f}")

    # 혼동 행렬 계산 및 시각화
    cm = confusion_matrix(y_test.numpy(), y_pred.numpy())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('SVM Confusion Matrix for MNIST Dataset')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    return model, cm

if __name__ == "__main__":
    model, confusion_mat = train_and_evaluate()

    # 주요 파라미터 설명
    print("\n주요 SVM 파라미터 설명:")
    print(f"1. 커널 함수 (Kernel): RBF 커널 - 비선형 경계 학습을 위한 가우시안 함수")
    print(f"2. 규제 파라미터 (C): {model.C} - 오분류에 대한 패널티 강도, 높을수록 하드 마진")
    print(f"3. 감마 (Gamma): {model.gamma} - RBF 커널의 폭 제어, 높을수록 결정 경계가 더 구불구불해짐")
    print(f"4. 최대 반복 횟수: {model.max_iter} - SMO 알고리즘의 최대 반복 횟수")