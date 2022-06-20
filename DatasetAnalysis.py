import json
import os
import random
import cv2
import matplotlib.pyplot as plt

color_map = [(244, 67, 54), (233, 30, 99), (156, 39, 176), (103, 58, 183), (63, 81, 181), (33, 150, 243), (3, 169, 244),
             (0, 188, 212), (0, 150, 136), (76, 175, 80)]

class CocoAnalysis:
    def __init__(self, anno_path, img_dir_path):
        print('init CocoAnalysis')
        self.img_dir_path = img_dir_path
        if os.path.isfile(anno_path):  # anno_path 파일이 있다면
            with open(anno_path, 'rt', encoding='UTF-8') as annotations:
                coco = json.load(annotations)
                self.info = coco['info']
                self.license = coco['licenses']
                self.images = coco['images']
                self.annotations = coco['annotations']
                self.categories = coco['categories']

                print('Total images: {}\nTotal boxes: {}\nTotal classes: {}'.format(len(self.images), len(self.annotations), len(self.categories)))
                self.class_names = [str(cate['id']) + ' : ' + cate['name'] for cate in self.categories]
                print('Class names : {} '.format(self.class_names))

                self.dic_imgid_annos = {}
                self.dic_imgid_img = {}
                for img in self.images:
                    self.dic_imgid_img[img['id']] = img
                    self.dic_imgid_annos[img['id']] = []

                self.dic_annoid_anno = {}
                for anno in self.annotations:
                    self.dic_imgid_annos[anno['image_id']].append(anno)
                    self.dic_annoid_anno[anno['id']] = anno

                self.dic_cateid_cate = {}
                for cate in self.categories:
                    self.dic_cateid_cate[cate['id']] = cate

        else:  # anno_path 파일이 없다면
            print('Check the annotation file path!!!')

    # random 으로 5개의 이미지를 바운딩 박스와 함께 출력
    def show_random_img_all(self):
        print_cnt = 5
        fig, axes = plt.subplots(1, print_cnt, figsize=(print_cnt ** 2, print_cnt))
        for o_idx in range(print_cnt):
            random_num = random.randint(0, len(self.images) - 1)
            img_instance = self.images[random_num]
            image = cv2.imread(self.img_dir_path + img_instance['file_name'].split('/')[-1])

            for a_idx, anno in enumerate(self.dic_imgid_annos[img_instance['id']]):
                bb = anno['bbox']
                image = cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                                      color_map[(anno['category_id'] + 1) % len(color_map)], 2)

                class_name = self.dic_cateid_cate[anno['category_id']]['name']

                #print(anno['image_id'], anno['id'], anno['category_id'], class_name)
                cv2.putText(image, class_name, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color_map[(anno['category_id'] + 1) % len(color_map)], 2)
            axes[o_idx].imshow(image[:, :, ::-1])
        plt.show()

    # random 으로 5개의 이미지를 특정 클래스 바운딩 박스와 함께 출력
    def show_random_img_class(self, class_num=0):
        print_cnt = 5
        fig, axes = plt.subplots(1, print_cnt, figsize=(print_cnt ** 2, print_cnt))
        for o_idx in range(print_cnt):
            random_num = random.randint(0, len(self.images) - 1)
            img_instance = self.images[random_num]
            image = cv2.imread(self.img_dir_path + img_instance['file_name'].split('/')[-1])

            for a_idx, anno in enumerate(self.dic_imgid_annos[img_instance['id']]):
                if class_num == anno['category_id']:
                    bb = anno['bbox']
                    image = cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                                          color_map[(anno['category_id'] + 1) % len(color_map)], 2)

                    class_name = self.dic_cateid_cate[anno['category_id']]['name']

                    #print(anno['image_id'], anno['id'], anno['category_id'], class_name)
                    cv2.putText(image, class_name, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                color_map[(anno['category_id'] + 1) % len(color_map)], 2)
            axes[o_idx].imshow(image[:, :, ::-1])
        plt.show()

    # 이미지 id에 대한 이미지와 바운딩 박스와 함께 출력
    def show_img_id(self, img_id=0):
        img_instance = self.dic_imgid_img[img_id]
        image = cv2.imread(self.img_dir_path + img_instance['file_name'].split('/')[-1])

        for a_idx, anno in enumerate(self.dic_imgid_annos[img_instance['id']]):
            bb = anno['bbox']
            image = cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                                  color_map[(anno['category_id'] + 1) % len(color_map)], 2)

            class_name = self.dic_cateid_cate[anno['category_id']]['name']

            #print(anno['image_id'], anno['id'], anno['category_id'], class_name)
            cv2.putText(image, class_name, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color_map[(anno['category_id'] + 1) % len(color_map)], 2)
        plt.imshow(image[:, :, ::-1])
        plt.show()


if __name__ == "__main__":
    anno_path = 'Self Driving Car.v3-fixed-small.coco/export/_annotations.coco.json'
    img_dir_path = 'Self Driving Car.v3-fixed-small.coco/export/'
    coco = CocoAnalysis(anno_path, img_dir_path)
    #coco.show_random_img_class()
    coco.show_img_id()
