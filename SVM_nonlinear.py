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


# MNIST 데이터 불러오기
def load_mnist_data():
    # MNIST 데이터셋 다운로드
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 학습 데이터 양 증가 (10000개)
    train_subset, _ = torch.utils.data.random_split(train_dataset, [10000, len(train_dataset) - 10000])
    
    train_loader = DataLoader(train_subset, batch_size=200, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    
    return train_loader, test_loader

# 커널 SVM 구현 (직접 구현)
class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=0.1, max_iter=100):
        self.kernel = kernel
        self.C = C  # 규제 파라미터 (소프트 마진)
        self.gamma = gamma  # RBF 커널의 파라미터
        self.max_iter = max_iter
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0.0  # 절편
    
    # 커널 함수 구현
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return torch.matmul(x1, x2.T)
        elif self.kernel == 'rbf':
            # RBF 커널: K(x,y) = exp(-gamma * ||x-y||^2)
            x1_norm = torch.sum(x1**2, dim=1).view(-1, 1)
            x2_norm = torch.sum(x2**2, dim=1).view(1, -1)
            dist_squared = x1_norm + x2_norm - 2.0 * torch.matmul(x1, x2.T)
            return torch.exp(-self.gamma * dist_squared)
        else:
            raise ValueError(f"지원하지 않는 커널: {self.kernel}")
    
    # SMO(Sequential Minimal Optimization) 알고리즘 간소화 버전
    def _smo_simplified(self, X, y):
        n_samples = X.shape[0]
        
        # 라그랑주 승수(알파) 초기화
        alpha = torch.zeros(n_samples, device=X.device)
        
        # 커널 행렬 계산
        K = self._kernel_function(X, X)
        
        # SMO 알고리즘 반복
        for iteration in range(self.max_iter):
            alpha_prev = alpha.clone()
            
            for i in range(n_samples):
                # 결정 함수 계산
                f_i = torch.sum(alpha * y * K[:, i]) + self.b
                
                # KKT 조건 확인 및 알파 업데이트
                if (y[i] * f_i < 1 and alpha[i] < self.C) or (y[i] * f_i > 1 and alpha[i] > 0):
                    # 두 번째 알파 선택 (여기서는 간단히 i+1 % n_samples)
                    j = (i + 1) % n_samples
                    
                    # eta 계산
                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 0:
                        continue
                    
                    # 알파_j 업데이트
                    alpha_j_old = alpha[j].clone()
                    alpha_i_old = alpha[i].clone()
                    
                    # L과 H 계산 (경계)
                    if y[i] == y[j]:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    else:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    
                    if L == H:
                        continue
                    
                    # 목적 함수의 2차 미분
                    f_i = torch.sum(alpha * y * K[:, i]) + self.b
                    f_j = torch.sum(alpha * y * K[:, j]) + self.b
                    E_i = f_i - y[i]
                    E_j = f_j - y[j]
                    
                    # 알파_j 업데이트
                    alpha[j] = alpha_j_old + y[j] * (E_i - E_j) / eta
                    
                    # 알파_j 클리핑
                    alpha[j] = torch.clamp(alpha[j], L, H)
                    
                    # 알파_i 업데이트
                    alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])
                    
                    # 절편 b 업데이트
                    b_i = self.b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b_j = self.b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < alpha[i] < self.C:
                        self.b = b_i
                    elif 0 < alpha[j] < self.C:
                        self.b = b_j
                    else:
                        self.b = (b_i + b_j) / 2
            
            # 수렴 확인
            diff = torch.norm(alpha - alpha_prev)
            if diff < 1e-3:
                print(f'SMO 알고리즘이 {iteration+1}번째 반복에서 수렴했습니다.')
                break
        
        # 서포트 벡터 찾기 (알파가 0보다 큰 포인트)
        sv_indices = torch.where(alpha > 1e-5)[0]
        self.alpha = alpha[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        
        return self
    
    # 학습 함수
    def fit(self, X, y):
        self._smo_simplified(X, y)
        return self
    
    # 예측 함수
    def predict(self, X):
        if self.alpha is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit 메소드를 호출하세요.")
        
        # 결정 함수 계산
        K = self._kernel_function(X, self.support_vectors)
        decision = torch.matmul(K, self.alpha * self.support_vector_labels) + self.b
        
        # 이진 분류의 경우 부호에 따라 클래스 결정
        return torch.sign(decision)

# 다중 클래스 분류를 위한 One-vs-Rest SVM
class MulticlassSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=0.1, max_iter=100):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.models = {}
    
    def fit(self, X, y):
        self.classes = torch.unique(y)
        
        # 각 클래스에 대해 OvR(One-vs-Rest) SVM 학습
        for cls in self.classes:
            print(f"클래스 {cls}에 대한 SVM 학습 중...")
            # 현재 클래스를 양성(+1), 나머지를 음성(-1)으로 레이블 변환
            binary_y = torch.where(y == cls, torch.tensor(1.0), torch.tensor(-1.0))
            
            # SVM 모델 학습
            model = KernelSVM(kernel=self.kernel, C=self.C, gamma=self.gamma, max_iter=self.max_iter)
            model.fit(X, binary_y)
            
            # 학습된 모델 저장
            self.models[cls.item()] = model
        
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        votes = torch.zeros((n_samples, len(self.classes)), device=X.device)
        
        # 각 클래스별 SVM의 결정 함수 값 계산
        for i, cls in enumerate(self.classes):
            model = self.models[cls.item()]
            
            # 결정 함수 값 계산
            K = model._kernel_function(X, model.support_vectors)
            decision = torch.matmul(K, model.alpha * model.support_vector_labels) + model.b
            
            # 결정 함수 값 저장
            votes[:, i] = decision
        
        # 가장 높은 결정 함수 값을 가진 클래스로 예측
        predictions = torch.argmax(votes, dim=1)
        return predictions

# 학습 및 평가 실행
def train_and_evaluate():
    # MNIST 데이터 로드
    train_loader, test_loader = load_mnist_data()
    
    # 학습 데이터셋에서 batch를 추출하여 학습에 사용
    X_train = []
    y_train = []
    for batch_idx, (data, target) in enumerate(train_loader):
        X_train.append(data.reshape(data.shape[0], -1))
        y_train.append(target)
        if batch_idx >= 4:  # 5개의 배치 사용 (약 1000개)
            break
    
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    
    # 테스트 데이터셋에서 batch를 추출하여 평가에 사용
    X_test = []
    y_test = []
    for batch_idx, (data, target) in enumerate(test_loader):
        X_test.append(data.reshape(data.shape[0], -1))
        y_test.append(target)
        if batch_idx >= 4:  # 5개의 배치 사용 (약 1000개)
            break
    
    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)
    
    # 다중 클래스 SVM 모델 생성 및 학습
    print("다중 클래스 SVM 모델 학습 중...")
    model = MulticlassSVM(kernel='rbf', C=10.0, gamma=0.001, max_iter=100)
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