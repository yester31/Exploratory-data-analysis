#https://universe.roboflow.com/roboflow-gw7yv/self-driving-car/3

from DatasetAnalysis import CocoAnalysis

if __name__ == "__main__":
    # annotation 파일 경로
    anno_path = 'Self Driving Car.v3-fixed-small.coco/export/_annotations.coco.json'
    # 실제 이미지가 위치한 폴더 경로
    img_dir_path = 'Self Driving Car.v3-fixed-small.coco/export/'

    # 객체 생성
    coco = CocoAnalysis(anno_path, img_dir_path)

    coco.show_random_img_all()

    coco.show_random_img_class(2)

    coco.show_img_id(0)