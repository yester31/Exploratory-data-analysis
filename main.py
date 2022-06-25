#https://universe.roboflow.com/roboflow-gw7yv/self-driving-car/3

from DatasetAnalysis import CocoAnalysis

if __name__ == "__main__":
    # annotation 파일 경로
    anno_path = 'Self Driving Car.v3-fixed-small.coco/export/_annotations.coco.json'
    # 실제 이미지가 위치한 폴더 경로
    img_dir_path = 'Self Driving Car.v3-fixed-small.coco/export/'

    # 객체 생성
    coco = CocoAnalysis(anno_path, img_dir_path)

    # 1. 이미지 표현
    # 랜덤 출력
    coco.show_random_img_all()
    # 램덤 출력 & 특정 클래스만 바운딩 박스 표시
    coco.show_random_img_class(2)
    # 특정 이미지 아이디의 이미지 출력
    coco.show_img_id(0)
    # 바운딩 박스 크기순으로 상위 5개 출력
    coco.show_big_annos()
    # 바운딩 박스 크기순으로 하위 5개 출력
    coco.show_small_annos()

    # 2. 차트 표현
    # 클래스 별 바운딩 박스의 개수를 바 차트로 출력
    coco.show_bar_chart_box_by_class()
    # 이미지 별로 바운딩 박스 개수를 박스 차트로 출력
    coco.show_bar_chart_box_by_image()
    # 이미지 별로 바운딩 박스 개수를 (내림차순으로) 박스 차트로 출력
    coco.show_bar_chart_box_by_image_descending()
    # 바운딩 박스의 가로 세로 길이를 산포도로 출력
    coco.show_scatter_chart_height_width_all()