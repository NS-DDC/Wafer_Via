def binarize_via(image, block_size=31, gray_drop=25.0, min_area=4, *,
                 shape_filter=True, max_aspect_ratio=1.8, min_circularity=0.65):
    """원본 한 장에서 어두운 VIA 후보를 0/255 마스크로 반환한다.

    입력:
        image: uint8 원본 배열. 회색조(H, W)/(H, W, 1), BGR(H, W, 3),
            BGRA(H, W, 4)를 자동으로 회색조로 처리한다. 컬러는 OpenCV BGR 순서.
            PIL 등의 RGB 배열은 호출 전에 cv2.COLOR_RGB2BGR로 변환한다.
        block_size: 주변 평균을 구할 정사각형 한 변(px). 3 이상의 홀수.
            실제 영상의 VIA 지름보다 충분히 크게, 우선 지름의 2~3배로 설정.
            너무 작으면 큰 VIA 내부가 빠지고, 너무 크면 주변 패턴이 섞인다.
        gray_drop: 주변 평균보다 얼마나 어두워야 하는지(0~255 밝기 단위).
            기본 25. 누락 시 15~20으로 낮추고, 잡음이 많으면 30~40으로 올린다.
        min_area: 연결된 흰 픽셀의 최소 면적(px). 기본 4는 1~3픽셀 잡음 제거.
            0이면 면적 필터만 끈다. 형상 필터까지 끄려면 shape_filter=False.
        shape_filter: 기본 True. 가늘고 긴 선과 원형에서 크게 벗어난 덩어리 제거.
        max_aspect_ratio: 회전 사각형의 긴 변/짧은 변 상한. 기본 1.8.
            낮추면 타원/선에 엄격, 높이면 길쭉한 VIA도 허용. 1 이상이어야 한다.
        min_circularity: 외곽 원형도 하한. 기본 0.65, 범위 0~1.
            원형도 = 4*pi*윤곽면적 / 윤곽둘레². 원은 1에 가깝고 불규칙하면 낮다.
            높이면 불규칙 잡음에 엄격, 낮추면 경계가 거친 VIA도 허용한다.
            작은 원은 픽셀 계단 때문에 1보다 낮으므로 처음부터 높게 잡지 않는다.

    반환:
        원본과 같은 (H, W) 크기의 uint8 배열. VIA 후보=255, 나머지=0.
        입력 배열을 수정하지 않으며 설계도, PAD 마스크, 중심 좌표는 필요 없다.

    원리:
        gray < 주변 block_size x block_size 평균 - gray_drop
        전체 영상 평균 대신 픽셀별 주변 평균을 써서 위치별 밝기 차이를 따른다.
        이진화 후 연결된 덩어리마다 면적, 회전 사각형 종횡비, 원형도를 검사한다.
        회전 사각형을 사용하므로 수평/수직뿐 아니라 대각선 선도 걸러낸다.
        통과한 덩어리는 원래 이진 픽셀 그대로 반환하며 원으로 다시 그리지 않는다.
        둥근 잡티처럼 VIA와 밝기/형상이 비슷한 것은 남을 수 있다. 원형도는
        완벽한 원 판별이 아니라 완만한 형상 필터이며, 설계 위치는 검사하지 않는다.
        영상 가장자리에서 잘렸거나 선과 붙은 VIA는 형상 필터로 빠질 수 있다.

    크기 기준:
        모든 길이/면적은 '실제 원본 영상'의 px/px² 기준이며 확대하지 않는다.
        설계도 가로/세로가 원본의 2배라면 설계도 VIA 지름 20px는 실측 약 10px.
        이 경우 block_size=31부터 시도한다. 설계도 면적은 약 1/4로 환산한다.

    사용:
        via_bin = binarize_via(image)
        via_bin = binarize_via(image, gray_drop=20, min_circularity=0.55)
        # 이전처럼 이진화만 수행 (면적/형상 필터 모두 OFF):
        via_bin = binarize_via(image, min_area=0, shape_filter=False)
    """
    # 함수 전체만 복사해도 사용할 수 있도록 의존성 import를 안에 둡니다.
    import cv2
    import numpy as np

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image는 비어 있지 않은 numpy 이미지 배열이어야 합니다.")
    if image.dtype != np.uint8:
        raise ValueError("image는 0~255 범위의 uint8 이미지여야 합니다.")
    if (not isinstance(block_size, (int, np.integer))
            or block_size < 3 or block_size % 2 == 0):
        raise ValueError("block_size는 3 이상의 홀수여야 합니다.")
    if not np.isfinite(gray_drop) or not 0 < gray_drop <= 255:
        raise ValueError("gray_drop은 0 초과 255 이하의 유한한 값이어야 합니다.")
    if not isinstance(min_area, (int, np.integer)) or min_area < 0:
        raise ValueError("min_area는 0 이상의 정수여야 합니다.")
    if not isinstance(shape_filter, (bool, np.bool_)):
        raise ValueError("shape_filter는 True 또는 False여야 합니다.")
    if not np.isfinite(max_aspect_ratio) or max_aspect_ratio < 1:
        raise ValueError("max_aspect_ratio는 1 이상의 유한한 값이어야 합니다.")
    if not np.isfinite(min_circularity) or not 0 <= min_circularity <= 1:
        raise ValueError("min_circularity는 0 이상 1 이하의 유한한 값이어야 합니다.")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        conversion = (cv2.COLOR_BGR2GRAY if image.shape[2] == 3
                      else cv2.COLOR_BGRA2GRAY)
        gray = cv2.cvtColor(image, conversion)
    else:
        raise ValueError("image는 회색조, BGR 또는 BGRA 이미지여야 합니다.")

    # [SECTOR: VIA_BINARIZATION_CORE] 이진화 기준을 직접 수정할 곳
    # float32로 계산해서 어두운 곳에서 임계값이 0 아래로 내려가도
    # uint8 뺄셈의 넘침(예: 10 - 25 -> 241)이 생기지 않게 합니다.
    gray_float = gray.astype(np.float32)
    local_mean = cv2.boxFilter(
        gray_float, ddepth=-1, ksize=(int(block_size), int(block_size)),
        normalize=True, borderType=cv2.BORDER_REPLICATE,
    )
    mask = (gray_float < local_mean - float(gray_drop)).astype(np.uint8) * 255
    # [END SECTOR: VIA_BINARIZATION_CORE]

    # [SECTOR: VIA_NOISE_FILTER] 작은 점 잡음과 선형/비원형 덩어리를 제거합니다.
    if min_area > 1 or shape_filter:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False  # 배경 라벨은 면적과 무관하게 0으로 유지합니다.

        if shape_filter:
            for label_id in range(1, count):
                if not keep[label_id]:
                    continue

                # 각 덩어리의 작은 사각 영역만 검사해 전체 영상을 반복 스캔하지 않습니다.
                x, y, w, h = stats[label_id, :4]
                blob = (labels[y:y + h, x:x + w] == label_id).astype(np.uint8)
                contours, _ = cv2.findContours(
                    blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    keep[label_id] = False
                    continue
                contour = max(contours, key=cv2.contourArea)

                # 1) 회전과 무관한 종횡비: 대각선도 가늘고 길면 제거합니다.
                (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
                short_side, long_side = sorted((side_a, side_b))
                if short_side <= 0 or long_side / short_side > max_aspect_ratio:
                    keep[label_id] = False
                    continue

                # 2) 외곽 원형도: 종횡비가 1에 가까워도 들쭉날쭉한 덩어리는 제거.
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4.0 * np.pi * area / (perimeter * perimeter)
                               if perimeter > 0 else 0.0)
                if area <= 0 or circularity < min_circularity:
                    keep[label_id] = False

        mask = keep[labels].astype(np.uint8) * 255
    # [END SECTOR: VIA_NOISE_FILTER]
    return mask
