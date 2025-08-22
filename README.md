# Fridge Classifier AI

Over the weekend of the AI Hackathon 2025, I prototyped a community food distribution system.

The idea is that there are community food pantries that are monitored by a camera which then gets processed by an AI image classifier to determine the fullness of the pantry, the type of food in it, and the safety of the food inside. This data would then be used to generate the most effeective routes for foodbanks to provide food for communities in need.

### FridgeClassifier

The FridgeClassifier prototype is a simple ResNet18 AI model that was trained on 100 images of fridges of various fullness. A similar AI could easily be trained on a dataset of captures from the community pantries.

* To use: run fridgeapp.py and open the link, which will open a browser tab with the front end where you can upload a photo of a fridge to classify.

* See the full shared repo including a route generation prototype [here](https://github.com/Lachyzzz1/KaiConnect)
