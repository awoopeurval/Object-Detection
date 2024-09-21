from ultralytics import YOLO
import cv2 as cv
import argparse
import time



class Objectdetection:

    def __init__(self, args) -> None:
        self.args = args

    def _image_object_detection(self, model):
        results = model(self.args.input)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv.imshow("YOLOv9 Inference", annotated_frame)
        # (this is necessary to avoid Python kernel form crashing)
        cv.waitKey(0)
        # Press q to close displayed window and stop the app
        if cv.waitKey(10) & 0xFF==ord('q'):
            # closing all open windows
            cv.destroyAllWindows()

    def _video_object_detection(self, model):
        cap = cv.VideoCapture(self.args.input)
        while cap.isOpened():
            # Read the frame from the video
            res, frame = cap.read()

            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

            if not res:
                print("Frame not found!!")
                break

            # Starting time for the process
            t1 = time.time()
            # Predict with the model
            results = model(frame)
            # Ending time for the process
            t2 = time.time()
            # Number of frames that appears within a second
            fps = 1/(t2 - t1)
            annotated_frame = results[0].plot()
            dim = (frame.shape[1], frame.shape[0])
            resized = cv.resize(annotated_frame, dim, interpolation = cv.INTER_AREA)
            # display the FPS
            cv.putText(resized, 'FPS : {:.2f}'.format(fps), (int((frame.shape[1] * 75) /100), 40), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (188, 205, 54), 2, cv.LINE_AA)
            # Display the annotated frame
            cv.imshow("YOLOv9 Inference", resized)

            # Break the loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Release the video capture object and close the display window
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Object Detection', description='Object detection with YOLOV9')
    parser.add_argument('-i', '--input', type=str, default='../media/images/intersection.jpg', 
                        help="path to video/image file to use as input")
    
    args = parser.parse_args()
    object_detection = Objectdetection(args)
    model = YOLO('yolov9c.pt')

    if args.input.endswith(('.jpg', '.jpeg', '.png')):
        object_detection._image_object_detection(model)
    else:
        object_detection._video_object_detection(model)