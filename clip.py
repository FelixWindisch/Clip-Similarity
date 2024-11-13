from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import torch
from torch.nn import CosineSimilarity
cossim = CosineSimilarity(dim=0, eps=1e-6)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
dataset = "blueberry"
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
original_images = {"blueberry" : Image.open("shared image Kopie.jpeg"), "tomato" : Image.open("Bildschirmfoto 2024-11-05 um 14.06.09.png")}
AI_images = {"blueberry" : ["1.43725_The scattered blueberries, just escaped the street_xl-1024-v1-0.png",
             "2.625355_The scattered blueberries on the cement sidewalk, _xl-1024-v1-0.png",
"3.187232_The scattered blueberries on the cement sidewalk, _xl-1024-v1-0.png",
             "4.362983_scattered blueberries on cement sidewalk, near str_xl-1024-v1-0.png"],
             "tomato" : ["2854_One rotten red tomato with a big hole on the front_xl-1024-v1-0.png",
                         "Firefly One rotten red tomato with a big hole on the front that is stained black at the edges, hangi.jpg"]}
prompts = {"blueberry" : ["The scattered blueberries, just escaped the street, besmirching the ground blueberry blue, surrounded by the gloomy road, the guiding system for blind people, the dirt captor cracks and brow tree seed.",
                            #"The scattered blueberries on the cement sidewalk, just escaped the street, besmirching the ground blueberry blue, surrounded by the gloomy road, the guiding system for blind people, the cracks in the ground and brown tree seeds.",
                            #"The scattered blueberries on the cement sidewalk, just escaped the street, besmirching the ground blue, surrounded by the gloomy road, the grooves and the cracks in the ground and brown tree seeds.",
                            #"scattered blueberries on cement sidewalk, near street,  surrounded by grey road, grooves and cracks in ground, brown tree seeds",
                            "A tree branch with green leaves extends over a clear, green-tinted body of water. The water is so transparent that the rocks and stones at the bottom are clearly visible, creating a calm and serene natural scene."
                            "A mossy branch leaning into the turquoise green water, stroking the shiny surface on which light reflects, beneath algae and stones are resting, above little branches with green leaves ramify.",
                            "A mossy branch from left down to right above leaning into the turquoise green water, stroking the shiny surface on which light reflections and litte water animals are dancing, beneath algae and stones are resting.",
                            "Fire"
                          ],
           "tomato" : ["One rotten red tomato with a big hole on the front that is stained black at the edges, hanging in the air on a green tomato plant with brown leaves, background is blurry green."]}
AI_images_loaded = [Image.open(x) for x in AI_images[dataset]]
def get_embeddings(imgs):
    inputs = processor(images=imgs, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features

def get_text_embeddings(txt):
    inputs = processor(text=txt, return_tensors="pt", padding=True, truncation=True)
    image_features = model.get_text_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features

original_embedding = get_embeddings([original_images[dataset]])[0]
AI_embeddings = get_embeddings(AI_images_loaded)
text_embeddings = get_text_embeddings(prompts[dataset])

print("Image Similarity")
for (index, AI_embedding) in enumerate(AI_embeddings):

    print(AI_images[dataset][index])
    print(cossim(original_embedding, AI_embedding))

print("Text Similarity")
for (index, text_embeddings) in enumerate(text_embeddings):
    print(prompts[dataset][index])
    print(cossim(original_embedding, text_embeddings))