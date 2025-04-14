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


# MNIST 데이터셋을 로드하는 함수
def load_mnist_data(train_samples=10000, batch_size=200):
    """
    MNIST 데이터셋을 로드하고 DataLoader 객체를 반환한다.
    
    Args:
        train_samples (int): 훈련에 사용할 샘플 수
        batch_size (int): 배치 크기
    
    Returns:
        tuple: (train_loader, test_loader) - 훈련 및 테스트 데이터 로더
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # PIL 이미지나 numpy 배열을 tensor로 변환
        transforms.Normalize((0.1307,), (0.3081,))  # 정규화 (평균, 표준편차)
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 훈련 데이터 부분집합 생성 (계산 효율성을 위해)
    train_subset, _ = torch.utils.data.random_split(
        train_dataset, [train_samples, len(train_dataset) - train_samples]
    )
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# 기본 SVM 클래스 (공통 기능 포함)
class BaseSVM:
    """
    SVM 분류기의 기본 클래스.
    
    선형 및 비선형 SVM 구현의 공통 기능을 제공한다.
    """
    
    def __init__(self, C=1.0, max_iter=100):
        """
        BaseSVM 분류기 초기화.
        
        Args:
            C (float): 규제 매개변수 (소프트 마진의 강도 조절)
            max_iter (int): SMO 알고리즘의 최대 반복 횟수
        """
        self.C = C  # 규제 파라미터 (소프트 마진 SVM에서 중요)
        self.max_iter = max_iter
        
        # 학습 후 설정될 속성들
        self.alpha = None  # 라그랑주 승수 (dual formulation에서 중요)
        self.support_vectors = None  # 서포트 벡터 (결정 경계를 정의하는 데이터 포인트)
        self.support_vector_labels = None  # 서포트 벡터의 레이블
        self.b = 0.0  # 절편 (bias term)
    
    def _smo_simplified(self, X, y, kernel_func):
        """
        SMO(Sequential Minimal Optimization) 알고리즘의 간소화 버전을 사용하여 SVM을 훈련한다.
        
        Args:
            X (torch.Tensor): 훈련 데이터 [n_samples, n_features]
            y (torch.Tensor): 훈련 레이블 [n_samples], 값은 +1 또는 -1
            kernel_func (callable): 커널 함수
            
        Returns:
            BaseSVM: 훈련된, 즉 자기 자신의 인스턴스
        """
        n_samples = X.shape[0]
        
        # 라그랑주 승수(알파) 초기화 - 모두 0으로 시작
        alpha = torch.zeros(n_samples, device=X.device)
        
        # 커널 행렬 계산 - 모든 훈련 샘플 쌍에 대한 커널 값
        # K[i,j] = kernel(x_i, x_j)
        K = kernel_func(X, X)  # [n_samples, n_samples]
        
        # SMO 알고리즘 반복
        for iteration in range(self.max_iter):
            alpha_prev = alpha.clone()  # 현재 반복의 알파 값 저장 (수렴 확인용)
            
            # 모든 훈련 샘플에 대해 반복
            for i in range(n_samples):
                # 샘플 i에 대한 결정 함수 값 계산
                # f_i = sum_j (alpha_j * y_j * K(x_j, x_i)) + b
                f_i = torch.sum(alpha * y * K[:, i]) + self.b
                
                # KKT(Karush-Kuhn-Tucker) 조건 확인 및 알파 업데이트
                # KKT 조건이 위반되면 알파를 업데이트해야 함
                if (y[i] * f_i < 1 and alpha[i] < self.C) or (y[i] * f_i > 1 and alpha[i] > 0):
                    # 두 번째 알파 선택 (여기서는 간단히 i+1 % n_samples)
                    j = (i + 1) % n_samples
                    
                    # eta = K(x_i, x_i) + K(x_j, x_j) - 2K(x_i, x_j)
                    # eta > 0이어야 알파 업데이트 가능 (목적 함수의 2차 미분)
                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 0:
                        continue
                    
                    # 알파_j의 현재 값 저장
                    alpha_j_old = alpha[j].clone()
                    alpha_i_old = alpha[i].clone()
                    
                    # L과 H 계산 (box constraint에 의한 경계)
                    # 알파_j의 새 값은 [L, H] 범위 내에 있어야 함
                    if y[i] == y[j]:
                        # 같은 레이블인 경우
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    else:
                        # 다른 레이블인 경우
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    
                    if L == H:
                        continue
                    
                    # 샘플 i와 j에 대한 오차 계산
                    # E_i = f_i - y_i (예측값과 실제값의 차이)
                    f_i = torch.sum(alpha * y * K[:, i]) + self.b
                    f_j = torch.sum(alpha * y * K[:, j]) + self.b
                    E_i = f_i - y[i]
                    E_j = f_j - y[j]
                    
                    # 알파_j 업데이트
                    # 알파_j_new = 알파_j_old - y_j * (E_i - E_j) / eta
                    alpha[j] = alpha_j_old + y[j] * (E_i - E_j) / eta
                    
                    # 알파_j를 [L, H] 범위로 클리핑
                    alpha[j] = torch.clamp(alpha[j], L, H)
                    
                    # 알파_i 업데이트 (알파_i와 알파_j의 합은 일정하게 유지)
                    # y_i * 알파_i + y_j * 알파_j = 상수 (제약 조건)
                    alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])
                    
                    # 절편 b 업데이트
                    # 두 가지 방법으로 b 계산 후 적절히 선택
                    b_i = self.b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b_j = self.b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    
                    # 0 < 알파 < C인 경우 해당 b 값 사용 (KKT 조건에 따라)
                    if 0 < alpha[i] < self.C:
                        self.b = b_i
                    elif 0 < alpha[j] < self.C:
                        self.b = b_j
                    else:
                        # 둘 다 경계에 있는 경우 평균 사용
                        self.b = (b_i + b_j) / 2
            
            # 수렴 확인: 알파 변화량이 임계값보다 작으면 수렴으로 간주
            diff = torch.norm(alpha - alpha_prev)
            if diff < 1e-3:
                print(f'SMO 알고리즘이 {iteration+1}번째 반복에서 수렴했습니다.')
                break
        
        # 서포트 벡터 찾기 (알파가 0보다 큰 포인트들)
        # 이들 포인트만 결정 경계에 영향을 미침
        sv_indices = torch.where(alpha > 1e-5)[0]
        self.alpha = alpha[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        
        return self


# 선형 SVM 구현
class LinearSVM(BaseSVM):
    """
    선형 SVM 분류기 클래스.
    
    선형 결정 경계를 학습하는 SVM 구현.
    """
    
    def __init__(self, C=1.0, max_iter=100):
        """
        LinearSVM 분류기 초기화.
        
        Args:
            C (float): 규제 매개변수 (소프트 마진의 강도 조절)
            max_iter (int): SMO 알고리즘의 최대 반복 횟수
        """
        super().__init__(C, max_iter)
        self.w = None  # 가중치 벡터 (선형 SVM에서만 사용)
    
    def _linear_kernel(self, x1, x2):
        """
        선형 커널 함수: K(x,y) = <x,y>
        
        Args:
            x1 (torch.Tensor): 첫 번째 데이터 포인트 집합 [n_samples1, n_features]
            x2 (torch.Tensor): 두 번째 데이터 포인트 집합 [n_samples2, n_features]
            
        Returns:
            torch.Tensor: 커널 행렬 [n_samples1, n_samples2]
        """
        return torch.matmul(x1, x2.T)
    
    def fit(self, X, y):
        """
        선형 SVM 모델을 훈련한다.
        
        Args:
            X (torch.Tensor): 훈련 데이터 [n_samples, n_features]
            y (torch.Tensor): 훈련 레이블 [n_samples], 값은 +1 또는 -1
            
        Returns:
            LinearSVM: 훈련된 모델 (자기 자신)
        """
        # SMO 알고리즘으로 모델 훈련
        super()._smo_simplified(X, y, self._linear_kernel)
        
        # 선형 SVM의 경우 가중치 벡터를 직접 계산할 수 있음
        # w = sum_i (alpha_i * y_i * x_i)
        self.w = torch.sum(
            self.alpha.view(-1, 1) * self.support_vector_labels.view(-1, 1) * self.support_vectors, 
            dim=0
        )
        
        return self
    
    def predict(self, X):
        """
        학습된 모델을 사용하여 새로운 데이터에 대한 예측을 수행한다.
        
        선형 SVM에서는 결정 함수를 직접 계산할 수 있다: f(x) = w·x + b
        
        Args:
            X (torch.Tensor): 예측할 데이터 [n_samples, n_features]
            
        Returns:
            torch.Tensor: 예측된 클래스 [n_samples], 값은 +1 또는 -1
            
        Raises:
            ValueError: 모델이 아직 훈련되지 않은 경우
        """
        if self.w is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit 메소드를 호출하세요.")
        
        # 선형 SVM의 결정 함수: f(x) = w·x + b
        # 내적을 계산한 후 절편 추가
        decision = torch.matmul(X, self.w) + self.b
        
        # 이진 분류의 경우 부호에 따라 클래스 결정 (sign 함수)
        return torch.sign(decision)


# 비선형 SVM (커널 SVM) 구현
class KernelSVM(BaseSVM):
    """
    커널 SVM 분류기 클래스.
    
    비선형 결정 경계를 학습할 수 있는 커널 함수를 사용하는 SVM 구현.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma=0.1, max_iter=100):
        """
        KernelSVM 분류기 초기화.
        
        Args:
            kernel (str): 사용할 커널 함수 ('linear' 또는 'rbf')
            C (float): 규제 매개변수 (소프트 마진의 강도 조절)
            gamma (float): RBF 커널의 매개변수 (결정 경계의 곡률 조절)
            max_iter (int): SMO 알고리즘의 최대 반복 횟수
        """
        super().__init__(C, max_iter)
        self.kernel = kernel
        self.gamma = gamma
    
    def _kernel_function(self, x1, x2):
        """
        두 데이터 포인트 집합 간의 커널 함수 값을 계산한다.
        
        Args:
            x1 (torch.Tensor): 첫 번째 데이터 포인트 집합 [n_samples1, n_features]
            x2 (torch.Tensor): 두 번째 데이터 포인트 집합 [n_samples2, n_features]
            
        Returns:
            torch.Tensor: 커널 행렬 [n_samples1, n_samples2]
            
        Raises:
            ValueError: 지원하지 않는 커널 유형일 경우
        """
        if self.kernel == 'linear':
            # 선형 커널: K(x,y) = <x,y> (내적)
            return torch.matmul(x1, x2.T)
        elif self.kernel == 'rbf':
            # RBF(Radial Basis Function) 커널: K(x,y) = exp(-gamma * ||x-y||^2)
            # 유클리드 거리의 제곱을 효율적으로 계산
            x1_norm = torch.sum(x1**2, dim=1).view(-1, 1)  # [n_samples1, 1]
            x2_norm = torch.sum(x2**2, dim=1).view(1, -1)  # [1, n_samples2]
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            dist_squared = x1_norm + x2_norm - 2.0 * torch.matmul(x1, x2.T)  # [n_samples1, n_samples2]
            return torch.exp(-self.gamma * dist_squared)
        else:
            raise ValueError(f"지원하지 않는 커널: {self.kernel}")
    
    def fit(self, X, y):
        """
        커널 SVM 모델을 훈련한다.
        
        Args:
            X (torch.Tensor): 훈련 데이터 [n_samples, n_features]
            y (torch.Tensor): 훈련 레이블 [n_samples], 값은 +1 또는 -1
            
        Returns:
            KernelSVM: 훈련된 모델 (자기 자신)
        """
        # SMO 알고리즘으로 모델 훈련
        super()._smo_simplified(X, y, self._kernel_function)
        return self
    
    def predict(self, X):
        """
        학습된 모델을 사용하여 새로운 데이터에 대한 예측을 수행한다.
        
        Args:
            X (torch.Tensor): 예측할 데이터 [n_samples, n_features]
            
        Returns:
            torch.Tensor: 예측된 클래스 [n_samples], 값은 +1 또는 -1
            
        Raises:
            ValueError: 모델이 아직 훈련되지 않은 경우
        """
        if self.alpha is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit 메소드를 호출하세요.")
        
        # 결정 함수 계산: f(x) = sum_i (alpha_i * y_i * K(x_i, x)) + b
        # 여기서 x_i는 서포트 벡터
        K = self._kernel_function(X, self.support_vectors)  # [n_samples, n_support_vectors]
        decision = torch.matmul(K, self.alpha * self.support_vector_labels) + self.b
        
        # 이진 분류의 경우 부호에 따라 클래스 결정 (sign 함수)
        return torch.sign(decision)


# 다중 클래스 SVM 기본 클래스
class MulticlassSVM:
    """
    다중 클래스 분류를 위한 One-vs-Rest SVM 분류기의 기본 클래스.
    
    각 클래스에 대해 개별 SVM 분류기를 학습시키고, 가장 높은 결정 함수 값을 가진
    클래스를 최종 예측으로 선택하는 전략을 사용한다.
    """
    
    def __init__(self, svm_constructor, **svm_params):
        """
        MulticlassSVM 분류기 초기화.
        
        Args:
            svm_constructor (class): 사용할 SVM 클래스 생성자 (LinearSVM 또는 KernelSVM)
            **svm_params: SVM 생성자에 전달할 매개변수들
        """
        self.svm_constructor = svm_constructor
        self.svm_params = svm_params
        self.models = {}  # 각 클래스별 SVM 모델을 저장할 딕셔너리
    
    def fit(self, X, y):
        """
        다중 클래스 SVM 모델을 훈련한다.
        
        각 클래스에 대해 One-vs-Rest 방식으로 이진 SVM 분류기를 학습시킨다.
        
        Args:
            X (torch.Tensor): 훈련 데이터 [n_samples, n_features]
            y (torch.Tensor): 훈련 레이블 [n_samples], 값은 정수 클래스 레이블
            
        Returns:
            MulticlassSVM: 훈련된 모델 (자기 자신)
        """
        # 고유한 클래스 레이블 식별
        self.classes = torch.unique(y)
        svm_type = "선형" if self.svm_constructor == LinearSVM else "비선형"
        
        # 각 클래스에 대해 OvR(One-vs-Rest) SVM 학습
        for cls in self.classes:
            print(f"클래스 {cls}에 대한 {svm_type} SVM 학습 중...")
            
            # 현재 클래스를 양성(+1), 나머지를 음성(-1)으로 레이블 변환
            # One-vs-Rest 전략의 핵심
            binary_y = torch.where(y == cls, torch.tensor(1.0), torch.tensor(-1.0))
            
            # 해당 클래스에 대한 이진 SVM 모델 학습
            model = self.svm_constructor(**self.svm_params)
            model.fit(X, binary_y)
            
            # 학습된 모델 저장 (클래스 레이블을 키로 사용)
            self.models[cls.item()] = model
        
        return self
    
    def predict(self, X):
        """
        학습된 모델을 사용하여 새로운 데이터에 대한 다중 클래스 예측을 수행한다.
        
        각 클래스별 SVM의 결정 함수 값을 계산하고, 가장 높은 값을 가진 클래스를 선택한다.
        
        Args:
            X (torch.Tensor): 예측할 데이터 [n_samples, n_features]
            
        Returns:
            torch.Tensor: 예측된 클래스 인덱스 [n_samples]
        """
        n_samples = X.shape[0]
        # 각 클래스에 대한 결정 함수 값을 저장할 텐서
        votes = torch.zeros((n_samples, len(self.classes)), device=X.device)
        
        # 각 클래스별 SVM의 결정 함수 값 계산
        for i, cls in enumerate(self.classes):
            model = self.models[cls.item()]
            
            # 내부 구현에 따라 예측 방식 변경
            if isinstance(model, LinearSVM):
                # 선형 SVM의 결정 함수: f(x) = w·x + b
                decision = torch.matmul(X, model.w) + model.b
            else:
                # 커널 SVM의 결정 함수: f(x) = sum_i (alpha_i * y_i * K(x_i, x)) + b
                K = model._kernel_function(X, model.support_vectors)
                decision = torch.matmul(K, model.alpha * model.support_vector_labels) + model.b
            
            # 결정 함수 값 저장
            votes[:, i] = decision
        
        # 가장 높은 결정 함수 값을 가진 클래스로 예측
        predictions = torch.argmax(votes, dim=1)
        return predictions


# 훈련 데이터 준비 함수
def prepare_data(train_loader, test_loader, num_batches=5):
    """
    데이터 로더에서 훈련 및 테스트 데이터를 추출하여 준비한다.
    
    Args:
        train_loader (DataLoader): 훈련 데이터 로더
        test_loader (DataLoader): 테스트 데이터 로더
        num_batches (int): 사용할 배치 수
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test) - 준비된 데이터
    """
    # 학습 데이터 추출
    X_train = []
    y_train = []
    for batch_idx, (data, target) in enumerate(train_loader):
        X_train.append(data.reshape(data.shape[0], -1))  # 이미지를 1차원 벡터로 변환
        y_train.append(target)
        if batch_idx >= (num_batches - 1):  # 지정된 배치 수만큼 사용
            break
    
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    
    # 테스트 데이터 추출
    X_test = []
    y_test = []
    for batch_idx, (data, target) in enumerate(test_loader):
        X_test.append(data.reshape(data.shape[0], -1))
        y_test.append(target)
        if batch_idx >= (num_batches - 1):
            break
    
    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)
    
    return X_train, y_train, X_test, y_test


# 모델 평가 및 시각화 함수
def evaluate_model(model, X_test, y_test, model_name="SVM", save_fig=True):
    """
    모델을 평가하고 혼동 행렬을 시각화한다.
    
    Args:
        model (MulticlassSVM): 평가할 SVM 모델
        X_test (torch.Tensor): 테스트 데이터
        y_test (torch.Tensor): 테스트 레이블
        model_name (str): 모델 이름 (파일명 및 타이틀용)
        save_fig (bool): 그림 저장 여부
        
    Returns:
        float: 모델 정확도
    """
    # 예측 수행
    y_pred = model.predict(X_test)
    
    # 정확도 계산
    accuracy = torch.sum(y_pred == y_test).float() / y_test.size(0)
    print(f"{model_name} 테스트 정확도: {accuracy:.4f}")
    
    # 혼동 행렬 계산 및 시각화
    cm = confusion_matrix(y_test.numpy(), y_pred.numpy())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} Confusion Matrix for MNIST Dataset')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy


# 선형 SVM과 커널 SVM 성능 비교 함수
def compare_svm_models(num_batches=5):
    """
    선형 SVM과 비선형(커널) SVM의 성능을 비교한다.
    
    Args:
        num_batches (int): 훈련에 사용할 배치 수
        
    Returns:
        tuple: (linear_model, kernel_model, linear_accuracy, kernel_accuracy)
    """
    # MNIST 데이터 로드
    train_loader, test_loader = load_mnist_data()
    
    # 데이터 준비
    X_train, y_train, X_test, y_test = prepare_data(train_loader, test_loader, num_batches)
    
    # 선형 SVM 모델 학습
    print("\n===== 선형 SVM 학습 =====")
    linear_model = MulticlassSVM(LinearSVM, C=10.0, max_iter=100)
    linear_model.fit(X_train, y_train)
    
    # 선형 SVM 평가
    linear_accuracy = evaluate_model(linear_model, X_test, y_test, "Linear SVM")
    
    # 커널 SVM 모델 학습
    print("\n===== 비선형 SVM (RBF 커널) 학습 =====")
    kernel_model = MulticlassSVM(KernelSVM, kernel='rbf', C=10.0, gamma=0.001, max_iter=100)
    kernel_model.fit(X_train, y_train)
    
    # 커널 SVM 평가
    kernel_accuracy = evaluate_model(kernel_model, X_test, y_test, "Nonlinear SVM (RBF Kernel)")
    
    # 결과 시각화 - 성능 비교 막대 그래프
    plt.figure(figsize=(10, 6))
    accuracies = [linear_accuracy.item(), kernel_accuracy.item()]
    plt.bar(['Linear SVM', 'Nonlinear SVM (RBF kernel)'], accuracies, color=['blue', 'orange'])
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Comparison of SVM model performance on MNIST dataset')
    plt.savefig('svm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 결과 출력
    print("\n===== SVM 모델 비교 결과 =====")
    print(f"선형 SVM 정확도: {linear_accuracy:.4f}")
    print(f"비선형 SVM 정확도: {kernel_accuracy:.4f}")
    print(f"정확도 차이: {(kernel_accuracy - linear_accuracy):.4f}")
    
    return linear_model, kernel_model, linear_accuracy, kernel_accuracy


# 메인 함수
def main():
    """
    메인 함수: 전체 프로그램 실행
    """
    # SVM 모델 비교
    linear_model, kernel_model, linear_acc, kernel_acc = compare_svm_models(num_batches=5)
    
    # 모델 특성 출력
    print("\n===== 선형 SVM 특성 =====")
    linear_svm = linear_model.models[0]  # 숫자 0에 대한 SVM 모델 선택
    print(f"가중치 벡터 크기: {linear_svm.w.shape}")
    print(f"서포트 벡터 수: {linear_svm.support_vectors.shape[0]}")
    print(f"규제 파라미터 (C): {linear_svm.C}")
    
    print("\n===== 비선형 SVM 특성 =====")
    kernel_svm = kernel_model.models[0]  # 숫자 0에 대한 SVM 모델 선택
    print(f"서포트 벡터 수: {kernel_svm.support_vectors.shape[0]}")
    print(f"규제 파라미터 (C): {kernel_svm.C}")
    print(f"커널: {kernel_svm.kernel}")
    print(f"감마 (RBF 커널): {kernel_svm.gamma}")
    
    # 모델 비교 설명
    print("\n===== 선형 SVM vs 비선형 SVM 차이점 =====")
    print("1. 결정 경계:")
    print("   - 선형 SVM: 데이터를 선형 초평면으로만 분리 가능")
    print("   - 비선형 SVM: 커널 트릭을 사용하여 복잡한 비선형 결정 경계 생성 가능")
    print("2. 계산 효율성:")
    print("   - 선형 SVM: 가중치 벡터(w)를 직접 계산하여 예측이 더 효율적")
    print("   - 비선형 SVM: 모든 서포트 벡터와의 커널 계산이 필요하여 예측이 더 느림")
    print("3. 모델 해석:")
    print("   - 선형 SVM: 특성의 가중치를 직접 확인 가능하여 더 해석하기 쉬움")
    print("   - 비선형 SVM: 암시적 특성 공간에서 작동하므로 해석이 어려움")
    print(f"4. MNIST 데이터셋에서의 성능:")
    print(f"   - 선형 SVM 정확도: {linear_acc:.4f}")
    print(f"   - 비선형 SVM 정확도: {kernel_acc:.4f}")
    print(f"   - 차이: {(kernel_acc - linear_acc):.4f} (비선형 SVM이 더 우수)")
    print("5. 서포트 벡터 수:")
    print(f"   - 선형 SVM: {linear_svm.support_vectors.shape[0]}개")
    print(f"   - 비선형 SVM: {kernel_svm.support_vectors.shape[0]}개")
    
    # MNIST 특성과 적합성
    print("\n===== MNIST 데이터셋과 SVM 적합성 =====")
    print("1. MNIST 데이터 특성:")
    print("   - 784차원(28x28 픽셀) 특성 공간")
    print("   - 숫자 간 비선형적인 차이 존재")
    print("   - 다양한 필체로 인한 클래스 내 분산")
    print("2. 선형 SVM 적합성:")
    print("   - 간단한 구현과 빠른 예측")
    print("   - 일부 숫자 쌍(예: 1과 7)은 선형으로 잘 분리됨")
    print("   - 복잡한 숫자 쌍(예: 3과 8)은 선형으로 분리하기 어려움")
    print("3. 비선형 SVM 적합성:")
    print("   - RBF 커널이 픽셀 간 지역적 관계를 잘 포착")
    print("   - 필체 변화에 대한 유연한 결정 경계 생성")
    print("   - 복잡한 숫자 패턴도 높은 정확도로 분류 가능")
    print(f"   - 성능 향상: {(kernel_acc - linear_acc)*100:.2f}% 개선")


if __name__ == "__main__":
    main()