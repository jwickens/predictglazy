from utils import load_raw_data, load_image_as_vec
from GlazeRecipes import GlazeRecipes
from GlazeMaterialDictionary import MaterialDictionary
from GlazeNet1 import Net
from flask import Flask, request, jsonify
import torch
import random
import string

app = Flask(__name__)

raw_data = load_raw_data()
material_dict = MaterialDictionary(raw_data)
recipes = GlazeRecipes(raw_data, material_dict)
model = Net(len(material_dict))
# model.load_state_dict(torch.load('model_state.pth'))
model.eval()


@app.route('/next', methods=['GET'])
def get_next_glaze():
    pass


@app.route('/image', methods=['GET'])
def get_image():
    pass


@app.route('/predict', methods=['POST'])
def predict():
    pass


@app.route('/train', methods=['POST'])
def train():
    pass


@app.route('/glaze', methods=['POST'])
def get_recipe_for_glaze():
    f = request.files['image']
    key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    f.save('/tmp/' + key)
    input_img = load_image_as_vec('/tmp/' + key)
    input_vector = torch.unsqueeze(input_img, 0)
    print(input_vector.shape)
    results = model(input_vector)
    output = results.squeeze()
    print(output.shape)
    return jsonify({
        'new': recipes.humanize_output(output),
        'closest': recipes.get_recipe_human(
            recipes.get_closest_recipe(output))
    })
