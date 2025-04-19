import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

def initialize_centroids(pixels, k, seed=42):
    """
    K-means 초기 중심점 무작위 초기화
    
    Args:
        pixels: 이미지 픽셀 데이터 (N×3)
        k: 중심점 개수
        seed: 랜덤 시드
        
    Returns:
        k개의 초기 중심점 (k×3)
    """
    random.seed(seed)
    # 전체 픽셀 인덱스 중에서 k개를 무작위로 샘플링
    idx = random.sample(range(pixels.shape[0]), k)
    # 샘플링된 인덱스에 해당하는 픽셀들을 초기 중심점으로 반환
    return pixels[idx]

def assign_clusters(pixels, centroids):
    """
    각 픽셀을 가장 가까운 중심점에 할당
    
    Args:
        pixels: 이미지 픽셀 데이터 (N×3)
        centroids: 현재 중심점 (k×3)
        
    Returns:
        각 픽셀의 클러스터 라벨 (N)
    """
    # 각 픽셀과 모든 중심점 간의 유클리드 거리 계산
    # pixels[:, None]은 (N, 1, 3) 형태로, centroids[None, :]은 (1, k, 3) 형태로 브로드캐스팅
    dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2) # 결과는 (N, k)
    # 각 픽셀에 대해 거리가 가장 짧은 중심점의 인덱스(클러스터 라벨)를 반환
    return np.argmin(dists, axis=1)

def recompute_centroids(pixels, labels, k):
    """
    클러스터별 픽셀들의 평균으로 새 중심점 계산
    
    Args:
        pixels: 이미지 픽셀 데이터 (N×3)
        labels: 픽셀별 클러스터 라벨 (N)
        k: 중심점 개수
        
    Returns:
        업데이트된 중심점 (k×3)
    """
    # 새로운 중심점을 저장할 배열 초기화
    centroids = np.zeros((k, 3), dtype=np.float32)
    # 각 클러스터(0부터 k-1까지)에 대해 반복
    for j in range(k):
        # 현재 클러스터(j)에 속하는 픽셀들만 선택
        cluster = pixels[labels == j]
        # 클러스터에 픽셀이 하나 이상 존재하면
        if len(cluster) > 0:
            # 해당 픽셀들의 RGB 평균값을 새로운 중심점으로 설정
            centroids[j] = cluster.mean(axis=0)
        # 만약 클러스터가 비어있다면, 중심점은 0으로 유지됨
    return centroids

def kmeans(pixels, k, max_iter=30, eps=1e-4):
    """
    K-means 알고리즘 실행
    
    Args:
        pixels: 이미지 픽셀 데이터 (N×3)
        k: 중심점 개수
        max_iter: 최대 반복 횟수
        eps: 수렴 판정 임계값 (중심점 변화량 기준)
        
    Returns:
        (중심점, 픽셀별 라벨) 튜플
    """
    # 초기 중심점 설정
    centroids = initialize_centroids(pixels, k)
    
    # 최대 반복 횟수만큼 또는 수렴할 때까지 반복
    for i in range(max_iter):
        # 1단계: 각 픽셀을 가장 가까운 중심점에 할당 (Assign)
        labels = assign_clusters(pixels, centroids)
        # 2단계: 클러스터별 평균으로 중심점 업데이트 (Update)
        new_centroids = recompute_centroids(pixels, labels, k)
        
        # 중심점 변화량이 임계값(eps)보다 작으면 수렴으로 간주하고 종료
        if np.allclose(centroids, new_centroids, atol=eps):
            print(f"K-means 수렴: {i+1}회 반복")
            break
            
        # 업데이트된 중심점으로 교체
        centroids = new_centroids
    
    # 최종 중심점과 각 픽셀의 클러스터 라벨 반환
    return centroids, labels

def load_image(path):
    """
    이미지 파일을 로드하고 RGB 형식의 numpy 배열로 변환
    
    Args:
        path: 이미지 파일 경로
        
    Returns:
        float32 타입의 H×W×3 형태 numpy 배열
    """
    # PIL 라이브러리를 사용하여 이미지 열기 및 RGB로 변환
    img = Image.open(path).convert('RGB')
    # 이미지를 float32 타입의 numpy 배열로 변환하여 반환
    return np.asarray(img, dtype=np.float32)

def save_image(arr, path):
    """
    numpy 배열을 이미지 파일로 저장
    
    Args:
        arr: 저장할 이미지 배열 (H×W×3)
        path: 저장할 파일 경로
    """
    # numpy 배열을 uint8 타입으로 변환 후 PIL Image 객체 생성 및 저장
    Image.fromarray(arr.astype(np.uint8)).save(path)

def compute_wcss(pixels, centroids, labels):
    """
    Within-Cluster Sum of Squares (WCSS) 계산
    
    Args:
        pixels: 이미지 픽셀 데이터 (N×3)
        centroids: 중심점 (k×3)
        labels: 픽셀별 클러스터 라벨 (N)
        
    Returns:
        WCSS 값
    """
    wcss = 0
    for j in range(len(centroids)):
        # 현재 클러스터에 속하는 픽셀들 선택
        cluster = pixels[labels == j]
        if len(cluster) > 0:
            # 클러스터 내의 각 픽셀과 중심점 간의 거리 제곱의 합 계산
            wcss += np.sum(np.linalg.norm(cluster - centroids[j], axis=1) ** 2)
    return wcss

def plot_wcss(pixels, k_list, out_dir='results'):
    """
    WCSS를 시각화하여 저장
    
    Args:
        pixels: 이미지 픽셀 데이터 (N×3)
        k_list: 시도할 k값 리스트
        out_dir: 결과 저장 디렉토리
    """
    wcss_values = []
    
    for k in k_list:
        print(f"k={k} WCSS 계산 중...")
        centroids, labels = kmeans(pixels, k)
        wcss = compute_wcss(pixels, centroids, labels)
        wcss_values.append(wcss)
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_list, wcss_values, marker='o')
    plt.title('WCSS vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    wcss_plot_path = os.path.join(out_dir, 'wcss_plot.png')
    plt.savefig(wcss_plot_path)
    plt.close()
    print(f'WCSS 그래프 저장 → {wcss_plot_path}')

def segment_image(img_arr, k_list, out_dir='results'):
    """
    이미지를 K-means로 분할하여 저장
    
    Args:
        img_arr: 원본 이미지 배열 (H×W×3)
        k_list: 시도할 k값 리스트
        out_dir: 결과 저장 디렉토리
    """
    os.makedirs(out_dir, exist_ok=True)
    h, w, c = img_arr.shape
    pixels = img_arr.reshape(-1, 3)
    
    # WCSS 그래프 생성
    plot_wcss(pixels, k_list, out_dir)
    
    for k in k_list:
        print(f"k={k} 클러스터링 시작...")
        centroids, labels = kmeans(pixels, k)
        segmented_img = centroids[labels].reshape(h, w, 3)
        out_path = os.path.join(out_dir, f'seg_k{k}.png')
        save_image(segmented_img, out_path)
        print(f'k={k} 분할 완료 → {out_path}')

def main():
    """
    메인 함수
    """
    if len(sys.argv) < 2:
        print("사용법: python kmeans.py tiger.jpg airplane.jpg")
        return
    
    k_values = [3, 6, 9]
    
    for img_path in sys.argv[1:]:
        try:
            print(f"이미지 처리: {img_path}")
            # 이미지 로드 함수 호출
            img = load_image(img_path)
            
            # 저장될 디렉토리 이름 설정 (원본 파일명 기반)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_dir = f'results_{base_name}'
            
            # 이미지 분할 함수 호출
            segment_image(img, k_values, out_dir=out_dir)
            print(f"이미지 {img_path} 처리 완료, 결과: {out_dir} 디렉토리")
        except Exception as e:
            # 오류 발생 시 메시지 출력
            print(f"이미지 {img_path} 처리 중 오류 발생: {e}")

if __name__ == '__main__':
    main()