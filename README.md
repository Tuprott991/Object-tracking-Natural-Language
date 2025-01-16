# README :>

This repo inherits the available modules from YOLOv9 for object detection phase. 
The above project proposes an object tracking method based on the vision language model (e.g., CLIP). The process divided into 4 phase. 
1. Natural Language query processing  (CLIP ViT B/32 text_encoder)
2. Candidates generating and bounding extracting (YOLOv9 Object detection)
3. Select the candidate that is most similar to the query (Cosine similarity, FAISS, L2 distance)
4. Candidates matching using DeepSort 