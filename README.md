# MushroomIdentifier

***Note: Mushroom identification from images alone is prone to errors, always consult a expert or guide if intending to consume mushrooms!***

MushroomIdentifier is a basic web app for identifying mushrooms! This project trains a MobileNetV2 model on the [2018 FGCVx Fungi Classification Challenge Dataset](https://www.kaggle.com/c/fungi-challenge-fgvc-2018/data) for identifying mushrooms based on images. 

## Running the App

First clone the repository and run `pip install requirements.txt` to download the necessary files and libraries. To run the app, simply run `python app.py`. In a web browser, navigate to `localhost:8000` or select the terminal output. 

The following screen should then display:


<p align="center">
  <img src=https://github.com/YouGoatMom/MushroomIdentifier/assets/173733073/1959b949-a676-4e07-ac46-70b1fbe0fdd2 width=500/>
</p>

Use the "Choose file" button to select an image, which will then display a preview on the page:


<p align="center">
  <img src=https://github.com/YouGoatMom/MushroomIdentifier/assets/173733073/1e10cd6b-1856-43fc-a280-07c988f8a6e3 width=500/>
</p>

Finally, select the submit button to view the top 5 results and corresponding model confidence:

<p align="center">
  <img src=https://github.com/YouGoatMom/MushroomIdentifier/assets/173733073/b9f5edee-fd36-44bf-989d-025b6784ac1c width=500/>
</p>

The top 5 results are intended to guide users to making a correct id. While the first id may be incorrect, web searching the additional results may provide the correct id or point users in the right direction. Have fun identifying mushrooms!

## The Model

The MobileNetV2 was trained on the [2018 FGCVx Fungi Classification Challenge Dataset](https://www.kaggle.com/c/fungi-challenge-fgvc-2018/data). If retraining the model or using a new architecture, the dataset directory is as follows:

```
project 
│
└───images
│   │
│   └───Abortiporus_biennis
│       │   img_1.txt
│       │   img_2.txt
│       │   ...
│   └───Achroomyces_disciformis
│       │   img_1.txt
│       │   img_2.txt
│       │   ...
|       ...
...
```

Run `python .\model\main.py` to train the model for 100 epochs. The model runs with an 80/20 train/validation split and the model is saved each epoch at `mushroom_id.pt`.
