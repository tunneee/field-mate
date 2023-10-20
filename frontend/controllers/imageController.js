import { ImageModel } from "../models/image.js";

export const createImage = async (req, res) => {
  try {
    const newImage = await ImageModel.create(req.body);
    res.status(201).json(newImage);
  } catch (error) {
    res.status(400).json({ error: 'Error creating image.' });
  }
};

export const getAllImages = async (req, res) => {
  try {
    const images = await ImageModel.find();
    res.status(200).json(images);
  } catch (error) {
    res.status(404).json({ error: 'Can not get all images' });
  }
};


export const getImageById = async (req, res) => {
  try {
    const image = await ImageModel.findById(req.params.id);
    res.status(200).json(image);
  } catch (error) {
    res.status(404).json({ error: 'Image not found.' });
  }
};
