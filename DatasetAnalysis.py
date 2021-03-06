import json
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns

color_map = [(244, 67, 54), (233, 30, 99), (156, 39, 176), (103, 58, 183), (63, 81, 181), (33, 150, 243), (3, 169, 244),
             (0, 188, 212), (0, 150, 136), (76, 175, 80)]


class CocoAnalysis:
    def __init__(self, anno_path, img_dir_path):
        print('init CocoAnalysis')
        if not os.path.exists('outputs'):  # 저장할 폴더가 없다면
            os.makedirs('outputs')  # 폴더 생성
            print('make directory {} is done'.format('outputs'))

        self.img_dir_path = img_dir_path
        if os.path.isfile(anno_path):  # anno_path 파일이 있다면
            with open(anno_path, 'rt', encoding='UTF-8') as annotations:
                coco = json.load(annotations)
                self.info = coco['info']
                self.license = coco['licenses']
                self.images = coco['images']
                self.annotations = coco['annotations']
                self.categories = coco['categories']

                print('Total images: {}\nTotal boxes: {}\nTotal classes: {}'.format(len(self.images),
                                                                                    len(self.annotations),
                                                                                    len(self.categories)))
                self.class_names = [cate['name'] for cate in self.categories]

                self.dic_imgid_anno_cnt = {}
                self.dic_imgid_annos = {}
                self.dic_imgid_img = {}
                for img in tqdm(self.images):
                    self.dic_imgid_anno_cnt[img['id']] = 0
                    self.dic_imgid_img[img['id']] = img
                    self.dic_imgid_annos[img['id']] = []


                self.dic_cateid_cate = {}
                for cate in tqdm(self.categories):
                    self.dic_cateid_cate[cate['id']] = cate

                self.boxes_by_class = np.zeros(len(self.categories))
                self.box_areas = np.zeros(len(self.annotations))
                self.box_heights = []
                self.box_widths = []
                for a_idx, anno in tqdm(enumerate(self.annotations)):
                    self.dic_imgid_anno_cnt[anno['image_id']] += 1
                    self.dic_imgid_annos[anno['image_id']].append(anno)
                    self.boxes_by_class[anno['category_id']] += 1
                    self.box_areas[a_idx] = anno['area']
                    self.box_widths.append(anno['bbox'][2])
                    self.box_heights.append(anno['bbox'][3])

                # 각 박스 사이즈 내림 정렬
                self.rev_area_idx = np.argsort(self.box_areas)[::-1]
                # 각 박스 사이즈 오름 정렬
                self.area_idx = np.argsort(self.box_areas)

        else:  # anno_path 파일이 없다면
            print('Check the annotation file path!!!')

    # visualization images ============================================

    # random 으로 5개의 이미지를 바운딩 박스와 함께 출력
    def show_random_img_all(self, save=False):
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

                # print(anno['image_id'], anno['id'], anno['category_id'], class_name)
                cv2.putText(image, class_name, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color_map[(anno['category_id'] + 1) % len(color_map)], 2)
            axes[o_idx].imshow(image[:, :, ::-1])
        if save:
            plt.savefig('./outputs/random_5_images.png', dpi=100)
        plt.show()

    # random 으로 5개의 이미지를 특정 클래스 바운딩 박스와 함께 출력
    def show_random_img_class(self, class_num=0, save=False):
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

                    # print(anno['image_id'], anno['id'], anno['category_id'], class_name)
                    cv2.putText(image, class_name, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                color_map[(anno['category_id'] + 1) % len(color_map)], 2)
            axes[o_idx].imshow(image[:, :, ::-1])

        if save:
            plt.savefig('./outputs/random_5_images_with_{}_anno.png'.format(class_name), dpi=100)
        plt.show()

    # 이미지 id에 대한 이미지와 바운딩 박스와 함께 출력
    def show_img_id(self, img_id=0, save=False):
        img_instance = self.dic_imgid_img[img_id]
        image = cv2.imread(self.img_dir_path + img_instance['file_name'].split('/')[-1])
        for a_idx, anno in enumerate(self.dic_imgid_annos[img_instance['id']]):
            bb = anno['bbox']
            image = cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                                  color_map[(anno['category_id'] + 1) % len(color_map)], 2)
            class_name = self.dic_cateid_cate[anno['category_id']]['name']
            # print(anno['image_id'], anno['id'], anno['category_id'], class_name)
            cv2.putText(image, class_name, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color_map[(anno['category_id'] + 1) % len(color_map)], 2)
        plt.imshow(image[:, :, ::-1])
        if save:
            plt.savefig('./outputs/image_id_{}.png'.format(img_id), dpi=100)
        plt.show()

    # 바운딩 박스 크기 상위 5개 출력
    def show_big_annos(self, save=False):
        print_cnt = 5
        fig, axes = plt.subplots(1, print_cnt, figsize=(print_cnt ** 2, print_cnt))
        for o_idx in range(print_cnt):
            anno_order = self.rev_area_idx[o_idx]
            anno = self.annotations[anno_order]
            img_id = anno['image_id']

            img_instance = self.dic_imgid_img[img_id]
            image = cv2.imread(self.img_dir_path + img_instance['file_name'].split('/')[-1])

            bb = anno['bbox']
            image = cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                                  color_map[(anno['category_id'] + 1) % len(color_map)], 2)

            class_name = self.dic_cateid_cate[anno['category_id']]['name']

            # print(anno['image_id'], anno['id'], anno['category_id'], class_name)
            cv2.putText(image, class_name, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color_map[(anno['category_id'] + 1) % len(color_map)], 2)
            axes[o_idx].imshow(image[:, :, ::-1])
        if save:
            plt.savefig('./outputs/big_anno_5_images.png', dpi=100)
        plt.show()

    # 바운딩 박스 크기 하위 5개 출력
    def show_small_annos(self, save=False):
        print_cnt = 5
        fig, axes = plt.subplots(1, print_cnt, figsize=(print_cnt ** 2, print_cnt))
        for o_idx in range(print_cnt):
            anno_order = self.area_idx[o_idx]
            anno = self.annotations[anno_order]
            img_id = anno['image_id']

            img_instance = self.dic_imgid_img[img_id]
            image = cv2.imread(self.img_dir_path + img_instance['file_name'].split('/')[-1])

            bb = anno['bbox']
            image = cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                                  color_map[(anno['category_id'] + 1) % len(color_map)], 2)

            class_name = self.dic_cateid_cate[anno['category_id']]['name']

            # print(anno['image_id'], anno['id'], anno['category_id'], class_name)
            cv2.putText(image, class_name, (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color_map[(anno['category_id'] + 1) % len(color_map)], 2)
            axes[o_idx].imshow(image[:, :, ::-1])
        if save:
            plt.savefig('./outputs/small_anno_5_images.png', dpi=100)
        plt.show()

    # visualization images ============================================

    # chart ============================================

    # 이미지 별로 바운딩 박스 개수를 박스 차트로 출력
    def show_bar_chart_box_by_image(self, save=False):
        image_list = self.dic_imgid_anno_cnt.items()
        x, y = zip(*image_list)
        plt.figure(figsize=(15, 10))
        colors = sns.color_palette('hls', len(self.images))
        plt.bar(np.arange(len(self.images)), y, width=1, color=colors)
        plt.yticks(np.arange(10) * max(self.dic_imgid_anno_cnt.values()) / 10,
                   np.arange(10) * int(max(self.dic_imgid_anno_cnt.values()) / 10), fontsize=15)
        plt.xticks(np.arange(10) * int(len(self.images) / 10), np.arange(10) * int(len(self.images) / 10), fontsize=15)
        plt.ylabel('The number of box', fontsize=15)
        plt.xlabel('Image', fontsize=15)
        plt.title('Distribution of boxes per image', fontsize=20)
        if save:
            plt.savefig('./outputs/bar_chart_box_by_image.png', dpi=100)
        plt.show()

    # (sub) 입력 value와 가장 근접한 값과 인덱스 반환
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    # 이미지 별로 바운딩 박스 개수를 (내림차순으로) 박스 차트로 출력
    def show_bar_chart_box_by_image_descending(self, save=False):
        rev_image_list = sorted(self.dic_imgid_anno_cnt.items(), reverse=True, key=lambda item: item[1])
        x, y = zip(*rev_image_list)
        # 평균 수직선 그리기
        box_count_avg = sum(self.dic_imgid_anno_cnt.values()) / len(self.dic_imgid_anno_cnt)
        _, nidx = self.find_nearest(y, box_count_avg)
        plt.figure(figsize=(15, 10))
        plt.vlines(nidx, 0, max(self.dic_imgid_anno_cnt.values()), color='red', linestyle='solid', linewidth=2)
        plt.text(nidx + 10, int(box_count_avg), 'avg box count({:.2f})'.format(box_count_avg), color='red', fontsize=15)
        # 바차트 그리기
        plt.bar(np.arange(len(self.images)), y, width=1)
        plt.yticks(np.arange(10) * max(self.dic_imgid_anno_cnt.values()) / 10,
                   np.arange(10) * int(max(self.dic_imgid_anno_cnt.values()) / 10), fontsize=15)
        plt.xticks(np.arange(10) * len(self.images) / 10,
                   ['00%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'], fontsize=15)
        plt.ylabel('The number of box', fontsize=15)
        plt.xlabel('Image', fontsize=15)
        plt.title('Distribution of boxes per image(descending order)', fontsize=20)
        if save:
            plt.savefig('./outputs/bar_chart_box_by_image_descending.png', dpi=100)
        plt.show()

    # 클래스 별로 바운딩 박스 개수를 박스 차트로 출력
    def show_bar_chart_box_by_class(self, save=False):
        plt.figure(figsize=(len(self.class_names), 10))

        colors = sns.color_palette('hls', len(self.class_names))  # 색상 맵 생성
        plt.bar(self.class_names, self.boxes_by_class, width=0.8, color=colors)
        plt.yticks(np.arange(10) * self.boxes_by_class.max() / 10, np.arange(10) * int(self.boxes_by_class.max() / 10),
                   fontsize=15)
        plt.xticks(np.arange(len(self.class_names)), self.class_names, fontsize=15, rotation=35, ha='right')
        plt.ylabel('The number of box', fontsize=15)
        plt.xlabel('class', fontsize=15)
        plt.title('Distribution of boxes per class', fontsize=20)
        for idx, class_cnt in enumerate(self.boxes_by_class):
            plt.text(idx, class_cnt, str(round(class_cnt / sum(self.boxes_by_class) * 100, 2)) + '%', fontsize=15,
                     color='red', horizontalalignment='center', verticalalignment='bottom')

        plt.tight_layout()  # 여백 조정
        plt.margins(x=0)  # 그래프 마진 지우기
        # plt.yscale("log") # y 축 스케일을 log로 변환
        if save:
            plt.savefig('./outputs/bar_chart_box_by_class.png', dpi=100)
        plt.show()

    # 바운딩 박스 height width 산포도 차트 출력
    def show_scatter_chart_height_width_all(self, save=False):
        plt.scatter(self.box_widths, self.box_heights, s=0.5)
        plt.vlines(256, 0, 512, color='red', linestyle='solid', linewidth=2)
        plt.hlines(256, 0, 512, color='red', linestyle='solid', linewidth=2)
        plt.ylabel('height', fontsize=15)
        plt.xlabel('width', fontsize=15)
        plt.title('Scatter distribution of box height & width', fontsize=20)
        #plt.tight_layout()  # 여백 조정
        #plt.margins(x=0, y=0)  # 그래프 마진 지우기
        if save:
            plt.savefig('./outputs/scatter_chart_height_width_all.png', dpi=100)
        plt.show()


    # chart ============================================

if __name__ == "__main__":
    anno_path = 'Self Driving Car.v3-fixed-small.coco/export/_annotations.coco.json'
    img_dir_path = 'Self Driving Car.v3-fixed-small.coco/export/'
    coco = CocoAnalysis(anno_path, img_dir_path)
    # coco.show_random_img_all()
    # coco.show_random_img_class()
    # coco.show_img_id()
    # coco.show_big_annos()
    # coco.show_small_annos()

    # coco.show_bar_chart_box_by_class(True)
    # coco.show_bar_chart_box_by_image()
    # coco.show_bar_chart_box_by_image_descending()
    # coco.show_scatter_chart_height_width_all()
