from deepface import DeepFace

"""
Face Verification. A one-to-one mapping of a given face against a known identity (e.g. is this the person?).
Face Identification. A one-to-many mapping for a given face against a database of known faces (e.g. who is this person?).
A face recognition system is expected to identify faces present in images and videos automatically. It can operate in either or both of two modes: (1) face verification (or authentication), and (2) face identification (or recognition).

"""
output_faces_directory = "./images/faces/face_15.jpg"
img2_path = "./images/faces2/face_45.jpg"
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
metric = ["cosine","euclidean", "euclidian_l2"]
detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]

verification = DeepFace.verify(img1_path = output_faces_directory, 
                               img2_path = img2_path,
                               model_name=models[1], #facenet
                               detector_backend=detectors[3], #dlib
                               enforce_detection=False,
                               distance_metric= metric[0]) #cosine

#print(verification)

"""
Verified: Indicates whether the faces in the two images are verified as the same person. In this case, it's True, meaning that the faces are verified.

distance: Represents the similarity distance between the embeddings of the two faces. Smaller distances generally indicate higher similarity. In this case, it's 0.14722980274053432.

threshold: This is the predefined threshold for face verification. If the distance is below this threshold, the faces are considered the same. In your result, the threshold is 0.68. Since your distance is below this threshold (0.1472), the faces are verified as the same person.

model: Specifies the face recognition model used. In your case, it's VGG-Face.

detector_backend: Indicates the face detection backend. Here, it's dlib.

similarity_metric: Specifies the metric used to measure the similarity between face embeddings. In your case, it's cosine.

facial_areas: Provides information about the detected facial areas in the images, including coordinates and dimensions.

time: Represents the time taken for the verification process.

"""
#Store images in a database, look for identity of image from database

#recognition = DeepFace.find(img_path = "img.jpg", db_path = "C:/facial_db")


#FACE ANALYSIS
analysis = DeepFace.analyze(img_path = output_faces_directory, detector_backend="dlib", 
                            enforce_detection=False, 
                            actions = ["age", "gender", "emotion", "race"]) 
print(analysis)


#Face recognition and attribute analysis in real-time videos
#DeepFace.stream(db_path = "C:/facial_db")



#DEEPFACE FACE EXTRACTION 
#img = DeepFace.detectFace("img1.jpg", detector_backend = detectors[3])

