import os
import face_recognition as fr

# encoded = {}
# for dirpath, dnames, fnames in os.walk("./faces"):
#         for d in dnames:
#             for dirpath, dnames, fnames in os.walk("./faces/" + d):
#                 count = 0
#                 name = d.split("/")[-1]
#                 encoded[name] = []
#                 for f in fnames:
#                     if f.endswith(".jpg") or f.endswith(".png"):
#                         face = fr.load_image_file("faces/" + d + "/" + f)
#                         encoding = fr.face_encodings(face)[0]
#                         encoded[name].append(encoding)
#             print("d")

def count_num_faces():
    dirpath, dnames, fnames = next(os.walk("./faces"))
    return len(fnames)


print(count_num_faces())