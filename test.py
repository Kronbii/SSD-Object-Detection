import cv2

config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
filename = "labels.txt"
with open(filename, "rt") as spt:
    classLabels = spt.read().rstrip("\n").split("\n")


model.setInputSize(
    320, 320
)  # greater this value better the reults but slower. Tune it for best results
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN

try:
    while True:

        ret, frame = cap.read()

        classIndex, confidence, bbox = model.detect(
            frame, confThreshold=0.65
        )  # tune the confidence  as required
        if len(classIndex) != 0 and len(classIndex) <81:
            for classInd, boxes in zip(classIndex.flatten(), bbox):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    classLabels[classInd - 1],
                    (boxes[0] + 10, boxes[1] + 40),
                    font,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2,
                )

        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except:
    cap.release()
    cv2.destroyAllWindows()
