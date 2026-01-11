import cv2
import numpy as np
import os

def compare_images(train_folder='./debug_train_samples', 
                   test_rois_folder='./debug_test_rois',
                   test_processed_folder='./debug_test_processed'):
    """학습 이미지와 테스트 이미지 비교"""
    
    # 학습 이미지 목록
    train_images = {f.split('_sample')[0]: f 
                    for f in os.listdir(train_folder) if f.endswith('.png')}
    
    # 테스트 이미지 입력
    test_image = input("비교할 테스트 이미지 파일명 입력 (예: slot_0_final.png): ")
    item_name = input("실제 아이템 이름 입력 (예: phoenix): ")
    
    if item_name not in train_images:
        print(f"'{item_name}' 학습 샘플을 찾을 수 없습니다.")
        print(f"사용 가능한 아이템: {list(train_images.keys())}")
        return
    
    # [수정] 파일명에서 폴더 자동 판단
    if 'original' in test_image or 'cropped' in test_image:
        test_folder = test_rois_folder
    else:  # 'final' 또는 그 외
        test_folder = test_processed_folder
    
    test_path = os.path.join(test_folder, test_image)
    train_path = os.path.join(train_folder, train_images[item_name])
    
    # 파일 존재 여부 확인
    if not os.path.exists(test_path):
        print(f"테스트 이미지를 찾을 수 없습니다: {test_path}")
        print(f"\n사용 가능한 파일:")
        for f in os.listdir(test_folder)[:20]:  # 처음 20개만
            print(f"  - {f}")
        return
    
    if not os.path.exists(train_path):
        print(f"학습 이미지를 찾을 수 없습니다: {train_path}")
        return
    
    # 이미지 로드
    train_img = cv2.imread(train_path)
    test_img = cv2.imread(test_path)
    
    if train_img is None:
        print(f"학습 이미지 로드 실패: {train_path}")
        return
    
    if test_img is None:
        print(f"테스트 이미지 로드 실패: {test_path}")
        return
    
    # 크기 맞추기
    h, w = 300, 300  # 200 -> 300으로 키움 (더 잘 보이게)
    train_resized = cv2.resize(train_img, (w, h))
    test_resized = cv2.resize(test_img, (w, h))
    
    # 나란히 배치
    gap = 20  # 간격
    comparison = np.ones((h, w*2 + gap, 3), dtype=np.uint8) * 255  # 흰 배경
    comparison[:, :w] = train_resized
    comparison[:, w+gap:] = test_resized
    
    # 텍스트 추가
    cv2.putText(comparison, "Training Sample", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "Test Sample", (w+gap+10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.putText(comparison, item_name, (10, h-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(comparison, test_image, (w+gap+10, h-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 차이 분석
    train_gray = cv2.cvtColor(train_resized, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(train_gray, test_gray)
    
    # 평균 색상 비교
    train_mean = cv2.mean(train_img)[:3]
    test_mean = cv2.mean(test_img)[:3]
    
    print(f"\n=== 차이 분석 ===")
    print(f"평균 픽셀 차이: {np.mean(diff):.2f} (0에 가까울수록 유사)")
    print(f"최대 픽셀 차이: {np.max(diff)}")
    print(f"\n학습 이미지 평균 색상(BGR): ({train_mean[0]:.1f}, {train_mean[1]:.1f}, {train_mean[2]:.1f})")
    print(f"테스트 이미지 평균 색상(BGR): ({test_mean[0]:.1f}, {test_mean[1]:.1f}, {test_mean[2]:.1f})")
    
    # 크기 비교
    print(f"\n학습 이미지 크기: {train_img.shape[:2]}")
    print(f"테스트 이미지 크기: {test_img.shape[:2]}")
    
    # 결과 표시
    cv2.imshow('Comparison (Press any key to close)', comparison)
    
    # 차이 맵도 표시
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    cv2.imshow('Difference Heatmap (Red=Big Diff)', diff_colored)
    
    # 결과 저장
    cv2.imwrite('./comparison_result.png', comparison)
    cv2.imwrite('./difference_heatmap.png', diff_colored)
    print("\n결과 이미지 저장:")
    print("  - comparison_result.png")
    print("  - difference_heatmap.png")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    compare_images()