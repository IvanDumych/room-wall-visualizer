# Room Wall Visualizer
A web application that allows the user to upload an image of a room and change the color of the walls or apply a selected texture.<br>

I am using here two pretrained models. One for wall segmentation and another for wall position estimation.<br>

This project utilizes code and a pretrained model for wall segmentation from the [OriginalProject](https://github.com/bjekic/WallSegmentation/).

This project utilizes code and a pretrained model for Indoor Scene Layout Estimation from the [OriginalProject](https://github.com/leVirve/lsun-room/).

## Model weights
Model weights was too large(600mb). You need to load it from drive and save at `wall_estimation/weight/`. <br>
Pre-trained weight at [Google Drive](https://drive.google.com/file/d/1aUJoXM9SQMe0LC38pA8v8r43pPOAaQ-a/view?usp=sharing).
## Getting Started

1. Create a virtual environment:

   ```
   python -m venv env
   ```

2. Start the virtual environment:

   ```
   env\Scripts\Activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
4. Run:

   ```
   python app.py
   ```
Open http://localhost:9000 <br>

Folder [test_images](./test_images) - contains images of rooms and textures to visualize the work
