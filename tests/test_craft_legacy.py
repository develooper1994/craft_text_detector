import unittest
import craft_text_detector

image_path = '../figures/idcard.png'
cuda = True  # False
show_time = False


class TestCraftTextDetector(unittest.TestCase):
    def test_load_craftnet_model(self):
        craft_net = craft_text_detector.load_craftnet_model()
        self.assertTrue(craft_net)

    def test_load_refinenet_model(self):
        refine_net = craft_text_detector.load_refinenet_model()
        self.assertTrue(refine_net)

    def test_get_prediction(self):
        # load image
        image = craft_text_detector.read_image(image_path)

        # load models
        craft_net = craft_text_detector.load_craftnet_model()
        refine_net = None

        # perform prediction
        text_threshold = 0.9
        link_threshold = 0.2
        low_text = 0.2
        get_prediction = craft_text_detector.get_prediction
        prediction_result = get_prediction(image=image,
                                           craft_net=craft_net,
                                           refine_net=refine_net,
                                           text_threshold=text_threshold,
                                           link_threshold=link_threshold,
                                           low_text=low_text,
                                           cuda=cuda,
                                           target_size=720,
                                           show_time=show_time)

        # !!! get_prediction.py -> get_prediction(...)
        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_LINEAR
        #     )
        # self.assertEqual(len(prediction_result["boxes"]), 35)

        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_CUBIC
        #     )
        self.assertEqual(37, len(prediction_result["boxes"]))
        self.assertEqual(4, len(prediction_result["boxes"][0]))
        self.assertEqual(2, len(prediction_result["boxes"][0][0]))
        self.assertEqual(111, int(prediction_result["boxes"][0][0][0]))
        # !!! get_prediction.py -> get_prediction(...)
        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_LINEAR
        #     )
        # self.assertEqual(len(prediction_result["polys"]), 35)

        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_CUBIC
        #     )
        self.assertEqual(37, len(prediction_result["polys"]))
        self.assertEqual((240, 368, 3), prediction_result["heatmaps"]["text_score_heatmap"].shape)

    def test_detect_text(self):
        prediction_result = craft_text_detector.detect_text(image=image_path, output_dir=None, rectify=True,
                                                            export_extra=False, text_threshold=0.7, link_threshold=0.4,
                                                            low_text=0.4, long_size=720, cuda=cuda, show_time=show_time,
                                                            refiner=False, crop_type="poly")
        # !!! get_prediction.py -> get_prediction(...)
        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_LINEAR
        #     )
        # self.assertEqual(len(prediction_result["boxes"]), 52)

        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_CUBIC
        #     )
        self.assertEqual(51, len(prediction_result["boxes"]))
        self.assertEqual(4, len(prediction_result["boxes"][0]))
        self.assertEqual(2, len(prediction_result["boxes"][0][0]))
        self.assertEqual(115, int(prediction_result["boxes"][0][0][0]))

        prediction_result = craft_text_detector.detect_text(image=image_path, output_dir=None, rectify=True,
                                                            export_extra=False, text_threshold=0.7, link_threshold=0.4,
                                                            low_text=0.4, long_size=720, cuda=cuda, show_time=show_time,
                                                            refiner=True, crop_type="poly")
        self.prediction_result_compare(prediction_result)

        prediction_result = craft_text_detector.detect_text(image=image_path, output_dir=None, rectify=False,
                                                            export_extra=False, text_threshold=0.7, link_threshold=0.4,
                                                            low_text=0.4, long_size=720, cuda=cuda, show_time=show_time,
                                                            refiner=False, crop_type="box")
        # !!! get_prediction.py -> get_prediction(...)
        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_LINEAR
        #     )
        # self.assertEqual(len(prediction_result["boxes"]), 52)

        #     img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        #         image, square_size, interpolation=cv2.INTER_CUBIC
        #     )
        self.assertEqual(51, len(prediction_result["boxes"]))
        self.assertEqual(4, len(prediction_result["boxes"][0]))
        self.assertEqual(2, len(prediction_result["boxes"][0][0]))
        self.assertEqual(244, int(prediction_result["boxes"][0][2][0]))

        prediction_result = craft_text_detector.detect_text(image=image_path, output_dir=None, rectify=False,
                                                            export_extra=False, text_threshold=0.7, link_threshold=0.4,
                                                            low_text=0.4, long_size=720, cuda=cuda, show_time=show_time,
                                                            refiner=True, crop_type="box")
        self.prediction_result_compare(prediction_result)

    def prediction_result_compare(self, prediction_result):
        self.assertEqual(19, len(prediction_result["boxes"]))
        self.assertEqual(4, len(prediction_result["boxes"][0]))
        self.assertEqual(2, len(prediction_result["boxes"][0][0]))
        self.assertEqual(661, int(prediction_result["boxes"][0][2][0]))


if __name__ == '__main__':
    unittest.main()
