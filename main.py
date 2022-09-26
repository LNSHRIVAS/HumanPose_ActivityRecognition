import cv2 

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

Pose_Pairs = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = 368
inHeight = 368

proto = "E:\Human_pose\Multi-pose-estimation\openpose\openpose-master\models\pose\mpi\pose_deploy_linevec_faster_4_stages.prototxt"
model = "E:\Human_pose\Multi-pose-estimation\openpose\openpose-master\models\pose\mpi\pose_iter_160000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, model)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True,        crop=False))

    out = net.forward()
    out = out[:, :19, :, :]
    assert(len(BODY_PARTS) == out.shape[1])
    points =[]

    for i in range(len(BODY_PARTS)):
    	# Slice heatmap of corresponging body's part.
    	heatMap = out[0, i, :, :]
    	# Originally, we try to find all the local maximums. To simplify a sample
    	# we just find a global one. However only a single pose at the same time
    	# could be detected this way.
    	_, conf, _, point = cv2.minMaxLoc(heatMap)
    	x = (frameWidth * point[0]) / out.shape[3]
    	y = (frameHeight * point[1]) / out.shape[2]
    # Add a point if it's confidence is higher than threshold.
    	points.append((int(x), int(y)) if conf > 0.2 else None)

    for pair in Pose_Pairs:
    	partFrom = pair[0]
    	partTo = pair[1]
    	assert(partFrom in BODY_PARTS)
    	assert(partTo in BODY_PARTS)
    	idFrom = BODY_PARTS[partFrom]
    	idTo = BODY_PARTS[partTo]
    	if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (255, 74, 0), 3)
            cv2.ellipse(frame, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            #cv.putText(frame, str(idFrom), points[idFrom], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
            #cv.putText(frame, str(idTo), points[idTo], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow("Image", frame)
    cv2.waitKey(1)
    
         
# Stop when the video is finished
cap.release()
     

    
	